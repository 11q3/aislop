// avmux — synthesize TTS, generate word-level ASS, burn subs, and mux with bgm/video.
// Adds XTTS support: -ttsLang and -ttsSpeakerWav are forwarded to Coqui TTS CLI.
//
// Build: go build -o avmux .
// Version inject: -ldflags "-X main.build=YYYYMMDDHHMMSS"
package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var build string // injected via -ldflags "-X main.build=YYYYMMDDHHMMSS"

func main() {
	// Required I/O
	video := flag.String("video", "", "background video file (required)")
	out := flag.String("out", "out.mp4", "output file")

	// Background music (required)
	music := flag.String("music", "", "background music file (required)")
	musicVol := flag.Float64("musicVol", 0.25, "linear gain for music (e.g. 0.25)")
	voiceVol := flag.Float64("voiceVol", 1.00, "linear gain for voice (e.g. 1.0)")
	musicLoop := flag.Bool("musicLoop", true, "loop background music to cover voice duration")

	// Randomized offsets
	videoStart := flag.Float64("videoStart", -1, "video start offset in seconds; -1 -> auto")
	musicStart := flag.Float64("musicStart", -1, "music start offset in seconds; -1 -> auto")
	randVideo := flag.Bool("randVideo", true, "randomize video start when -videoStart < 0")
	randMusic := flag.Bool("randMusic", true, "randomize music start when -musicStart < 0")
	seed := flag.Int64("seed", 0, "PRNG seed; 0 -> time-based")

	timeout := flag.Duration("timeout", 0, "overall timeout (e.g. 5m)")

	// NVENC
	useGPU := flag.Bool("useGPU", false, "use NVIDIA NVENC")
	gpuPreset := flag.String("gpuPreset", "p1", "NVENC preset p1..p7 (p7=slow)")
	gpuRC := flag.String("gpuRC", "vbr_hq", "NVENC rc: vbr|vbr_hq|constqp")
	gpuCQ := flag.String("gpuCQ", "19", "quality: vbr/vbr_hq -> -cq, constqp -> -qp (0..51)")

	// Subtitles (always generate + burn)
	assOut := flag.String("assOut", "", "where to write the generated ASS (default: next to -out)")
	py := flag.String("python", ".venv/bin/python", "python executable to run the generator")
	pyScript := flag.String("pyScript", "scripts/make_ass_words.py", "subtitle generator script")
	whModel := flag.String("whisperModel", "small", "faster-whisper model")
	whCompute := flag.String("whisperCompute", "float16", "float16|int8_float16|float32")

	// TTS (always synthesize from story file)
	ttsBin := flag.String("ttsBin", "/home/elevenqtwo/TTS/.venv311/bin/tts", "path to `tts` CLI")
	storyFile := flag.String("storyFile", "", "UTF-8 text file to synthesize (required)")
	voiceOut := flag.String("voiceOut", "story.wav", "output WAV from TTS (becomes voice track)")
	ttsModel := flag.String("ttsModel", "tts_models/en/vctk/vits", "Coqui TTS model_name")
	ttsSpeaker := flag.String("ttsSpeaker", "p376", "speaker id/index or name")
	ttsSpeakerWav := flag.String("ttsSpeakerWav", "", "reference WAV for XTTS cloning")
	ttsLang := flag.String("ttsLang", "", "language idx for XTTS (en, ru, ja, ...)")
	ttsCUDA := flag.Bool("ttsCUDA", true, "pass --use_cuda true/false to tts")

	// Utility
	debug := flag.Bool("debug", false, "print parsed flags and decisions")
	version := flag.Bool("version", false, "print version and exit")

	flag.Parse()

	if *version {
		if build == "" {
			fmt.Println("avmux (dev)")
		} else {
			fmt.Println("avmux", build)
		}
		return
	}

	must(ensureInPath("ffmpeg"), "ffmpeg not in PATH")
	must(ensureInPath("ffprobe"), "ffprobe not in PATH")

	// Required inputs present + exist
	if *video == "" || !pathExists(*video) {
		fail("no background video")
	}
	if *music == "" || !pathExists(*music) {
		fail("no background music")
	}
	if *out == "" {
		fail("output path missing")
	}
	if *storyFile == "" || !pathExists(*storyFile) {
		fail("no story text")
	}

	// TTS: always synthesize from story file
	if _, err := os.Stat(*ttsBin); err != nil {
		fail("tts not found at %s: %v", *ttsBin, err)
	}
	b, err := os.ReadFile(*storyFile)
	must(err, "read story file failed: %v", err)
	text := strings.TrimSpace(string(b))
	if text == "" {
		fail("no story text")
	}
	_ = os.Remove(*voiceOut) // ensure fresh synth
	if err := runTTS(*ttsBin, text, *ttsModel, *ttsSpeaker, *ttsSpeakerWav, *ttsLang, *ttsCUDA, *voiceOut, *timeout); err != nil {
		fail("unable to merge video+speech")
	}
	voicePath := *voiceOut

	// durations
	audDur, err := probeDuration(voicePath)
	must(err, "probe voice duration failed")
	vidDur, err := probeDuration(*video)
	must(err, "probe video duration failed")
	musicDur, err := probeDuration(*music)
	must(err, "probe music duration failed")

	// PRNG
	if *seed != 0 {
		rand.Seed(*seed)
	} else {
		rand.Seed(time.Now().UnixNano())
	}

	// Decide randomized starts
	vStart := *videoStart
	if vStart < 0 {
		if *randVideo {
			if audDur <= vidDur {
				vStart = randRange(0, maxf(vidDur-audDur, 0))
			} else {
				vStart = randRange(0, vidDur) // will loop
			}
		} else {
			vStart = 0
		}
	}
	mStart := *musicStart
	if mStart < 0 {
		if *randMusic {
			if *musicLoop && audDur > musicDur {
				mStart = randRange(0, musicDur) // will loop
			} else {
				mStart = randRange(0, maxf(musicDur-audDur, 0))
			}
		} else {
			mStart = 0
		}
	}

	if *debug {
		fmt.Println("== parsed flags ==")
		fmt.Printf("  -video=%q\n", *video)
		fmt.Printf("  -music=%q\n", *music)
		fmt.Printf("  -musicVol=%.3f -voiceVol=%.3f -musicLoop=%v\n", *musicVol, *voiceVol, *musicLoop)
		fmt.Printf("  -out=%q\n", *out)
		fmt.Printf("  -assOut=%q\n", *assOut)
		fmt.Printf("  -python=%q\n", *py)
		fmt.Printf("  -pyScript=%q\n", *pyScript)
		fmt.Printf("  -whisperModel=%q\n", *whModel)
		fmt.Printf("  -whisperCompute=%q\n", *whCompute)
		fmt.Printf("  -ttsBin=%q\n", *ttsBin)
		fmt.Printf("  -ttsModel=%q\n", *ttsModel)
		fmt.Printf("  -ttsSpeaker=%q\n", *ttsSpeaker)
		fmt.Printf("  -ttsSpeakerWav=%q\n", *ttsSpeakerWav)
		fmt.Printf("  -ttsLang=%q\n", *ttsLang)
		fmt.Printf("  -ttsCUDA=%v\n", *ttsCUDA)
		fmt.Printf("  -timeout=%q\n", *timeout)
		fmt.Printf("  voice: %.3fs, video: %.3fs, music: %.3fs\n", audDur, vidDur, musicDur)
		fmt.Printf("  seeds: seed=%d randVideo=%v randMusic=%v\n", *seed, *randVideo, *randMusic)
		fmt.Printf("  chosen offsets: videoStart=%.3fs musicStart=%.3fs\n", vStart, mStart)
		fmt.Println("===================")
	}

	// Decide ASS path (always generate + burn)
	finalASS := *assOut
	if finalASS == "" {
		outDir := filepath.Dir(*out)
		outBase := strings.TrimSuffix(filepath.Base(*out), filepath.Ext(*out))
		finalASS = filepath.Join(outDir, outBase+".ass")
	}

	// Generate word-level ASS from voice; device always cuda
	must(ensureCallable(*py, "--version"), "python not callable: %s", *py)
	assDir := filepath.Dir(finalASS)
	tmpName := "subs.ass"
	tmpASS := filepath.Join(assDir, tmpName)
	_ = os.Remove(tmpASS)
	_ = os.Remove(finalASS)

	env := append(os.Environ(),
		"WHISPER_MODEL="+*whModel,
		"WHISPER_COMPUTE="+*whCompute,
		"DEVICE=cuda",
	)
	cmd := exec.Command(*py, *pyScript, voicePath)
	cmd.Env = env
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Dir = assDir // script writes subs.ass in its CWD
	if err := cmd.Run(); err != nil {
		fail("unable to generate subtitles")
	}
	if !pathExists(tmpASS) {
		fail("unable to generate subtitles")
	}
	must(os.Rename(tmpASS, finalASS), "rename %s -> %s failed", tmpASS, finalASS)
	absAss, _ := filepath.Abs(finalASS)
	assPath := absAss

	// Single-pass final mux with randomized offsets
	if err := muxVideoVoiceMusic(
		*video, voicePath, *music, assPath, *out, *timeout,
		*useGPU, *gpuPreset, *gpuRC, *gpuCQ,
		audDur, vidDur, musicDur,
		*musicVol, *voiceVol, *musicLoop,
		vStart, mStart,
	); err != nil {
		fail("unable to merge video+background music")
	}

	fmt.Println("done:", *out)
}

func muxVideoVoiceMusic(
	video, voice, music, ass, out string, to time.Duration,
	useGPU bool, gpuPreset, gpuRC, gpuCQ string,
	audDur, vidDur, musicDur float64,
	musicVol, voiceVol float64, musicLoop bool,
	videoStart, musicStart float64,
) error {
	args := []string{"-y"}

	// Video input (seek + optional loop)
	if audDur > vidDur {
		args = append(args, "-stream_loop", "-1") // applies to next input (video)
	}
	args = append(args, "-ss", fmtSec(videoStart), "-i", video)

	// Voice input (no seek)
	args = append(args, "-i", voice)

	// Music input (optional loop + seek)
	if musicLoop && audDur > musicDur {
		args = append(args, "-stream_loop", "-1")
	}
	args = append(args, "-ss", fmtSec(musicStart), "-i", music)

	// burn ASS
	args = append(args, "-vf", "ass="+ass)

	// limit to voice length
	args = append(args, "-t", fmtSec(audDur))

	// audio mixing
	af := fmt.Sprintf(
		"[1:a]volume=%g,aresample=async=1:first_pts=0,aformat=sample_rates=44100:channel_layouts=stereo[v];"+
			"[2:a]volume=%g,aresample=async=1:first_pts=0,aformat=sample_rates=44100:channel_layouts=stereo[m];"+
			"[v][m]amix=inputs=2:duration=first:dropout_transition=0,aresample=async=1[aout]",
		voiceVol, musicVol,
	)
	args = append(args, "-filter_complex", af, "-map", "0:v:0", "-map", "[aout]")

	// encoder
	if useGPU && hasEncoder("h264_nvenc") {
		args = append(args, "-c:v", "h264_nvenc", "-preset", *gpuPreset, "-pix_fmt", "yuv420p")
		switch strings.ToLower(*gpuRC) {
		case "constqp":
			args = append(args, "-rc", "constqp", "-qp", *gpuCQ)
		case "vbr":
			args = append(args, "-rc", "vbr", "-cq", *gpuCQ, "-b:v", "0")
		default:
			args = append(args, "-rc", "vbr_hq", "-cq", *gpuCQ, "-b:v", "0", "-tune", "hq")
		}
	} else {
		args = append(args, "-c:v", "libx264", "-preset", "veryfast", "-crf", *gpuCQ, "-pix_fmt", "yuv420p")
	}

	// audio + container flags
	args = append(args, "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out)

	return runFFmpegErr(args, to)
}

// --- helpers ---

func runTTS(ttsBin, text, model, speaker, speakerWav, lang string, useCUDA bool, outPath string, to time.Duration) error {
	args := []string{
		"--text", text,
		"--model_name", model,
		"--out_path", outPath,
	}
	if speaker != "" {
		args = append(args, "--speaker_idx", speaker)
	}
	if speakerWav != "" {
		args = append(args, "--speaker_wav", speakerWav)
	}
	if lang != "" {
		args = append(args, "--language_idx", lang)
	}
	if useCUDA {
		args = append(args, "--use_cuda", "true")
	} else {
		args = append(args, "--use_cuda", "false")
	}

	fmt.Printf("running: %s %s\n", ttsBin, strings.Join(quote(args), " "))
	var ctx context.Context
	var cancel func()
	if to > 0 {
		ctx, cancel = context.WithTimeout(context.Background(), to)
	} else {
		ctx, cancel = context.WithCancel(context.Background())
	}
	defer cancel()

	cmd := exec.CommandContext(ctx, ttsBin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return fmt.Errorf("tts timed out after %v", to)
		}
		return err
	}
	if _, err := os.Stat(outPath); err != nil {
		return fmt.Errorf("tts did not produce %s", outPath)
	}
	return nil
}

func runFFmpegErr(args []string, to time.Duration) error {
	fmt.Printf("running: ffmpeg %s\n", strings.Join(quote(args), " "))
	var ctx context.Context
	var cancel func()
	if to > 0 {
		ctx, cancel = context.WithTimeout(context.Background(), to)
	} else {
		ctx, cancel = context.WithCancel(context.Background())
	}
	defer cancel()
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return fmt.Errorf("ffmpeg timed out after %v", to)
		}
		return err
	}
	return nil
}

func ensureInPath(bin string) error {
	cmd := exec.Command(bin, "-version")
	var buf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &buf, &buf
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s not callable: %w\n%s", bin, err, buf.String())
	}
	return nil
}

func ensureCallable(bin string, arg string) error {
	cmd := exec.Command(bin, arg)
	var buf bytes.Buffer
	cmd.Stdout, cmd.Stderr = &buf, &buf
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s %s failed: %w\n%s", bin, arg, err, buf.String())
	}
	return nil
}

func probeDuration(path string) (float64, error) {
	cmd := exec.Command("ffprobe",
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		path,
	)
	out, err := cmd.Output()
	if err != nil {
		return 0, err
	}
	s := strings.TrimSpace(string(out))
	sec, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("parse duration %q: %w", s, err)
	}
	if sec < 0 {
		return 0, fmt.Errorf("negative duration: %s", path)
	}
	return sec, nil
}

func quote(s []string) []string {
	res := make([]string, len(s))
	for i, v := range s {
		if strings.ContainsAny(v, " \t\"'") {
			res[i] = strconv.Quote(v)
		} else {
			res[i] = v
		}
	}
	return res
}

func hasEncoder(name string) bool {
	out, err := exec.Command("ffmpeg", "-hide_banner", "-encoders").Output()
	if err != nil {
		return false
	}
	want := strings.ToLower(strings.TrimSpace(name))
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && strings.ToLower(fields[1]) == want {
			return true
		}
	}
	return false
}

func pathExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func must(err error, format string, a ...any) {
	if err != nil {
		fail(format, a...)
	}
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", a...)
	os.Exit(1)
}

func randRange(min, max float64) float64 {
	if max <= min {
		return min
	}
	return min + rand.Float64()*(max-min)
}

func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func fmtSec(f float64) string {
	return fmt.Sprintf("%.3f", f)
}
