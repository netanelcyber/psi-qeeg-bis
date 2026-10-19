"""Clock, causal pairing, media I/O and report tests; synthetic fixtures only."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from psi_qeeg.cli import main
from psi_qeeg.recording import Recording
from psi_qeeg.video_eeg import VideoClock, frame_difference, probe_video, read_annotations, _raw_display, write_video_eeg_report


class ClockTests(unittest.TestCase):
    def test_known_offset_drift_and_inverse(self):
        clock = VideoClock.fit([{"video_s":v,"eeg_s":100+1.0001*v} for v in (0,50,100)], 1)
        self.assertAlmostEqual(clock.rate, 1.0001, places=10)
        self.assertAlmostEqual(clock.offset_s, 100, places=10)
        np.testing.assert_allclose(clock.to_video(clock.to_eeg([0,25,100])), [0,25,100], atol=1e-10)

    def test_bad_or_inconsistent_anchors_rejected(self):
        cases = [[], [{"video_s":0,"eeg_s":100}],
                 [{"video_s":0,"eeg_s":100},{"video_s":0,"eeg_s":101}],
                 [{"video_s":0,"eeg_s":100},{"video_s":1,"eeg_s":99}],
                 [{"video_s":0,"eeg_s":100},{"video_s":1,"eeg_s":float("nan")}],
                 [{"video_s":0,"eeg_s":100},{"video_s":1,"eeg_s":102},{"video_s":2,"eeg_s":102}]]
        for anchors in cases:
            with self.subTest(anchors=anchors), self.assertRaises(ValueError):
                VideoClock.fit(anchors, 1)
        with self.assertRaisesRegex(ValueError, "residual"):
            VideoClock.fit([{"video_s":0,"eeg_s":100},{"video_s":1,"eeg_s":101.2},{"video_s":2,"eeg_s":102}], 1)

    def test_overlap_clips_without_extrapolating_video(self):
        clock = VideoClock(1,100,0,1)
        self.assertEqual(clock.overlap(200,260,180), (100,160))
        with self.assertRaisesRegex(ValueError,"no overlapping"):
            clock.overlap(300,400,180)

    def test_annotations_require_clock_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"events.csv"
            rows=[{"start_s":1,"end_s":2,"clock":"video","label":"Test movement","source":"synthetic_script"}]
            pd.DataFrame(rows).to_csv(p,index=False)
            events=read_annotations(p,VideoClock(1,100,0,1),100,110)
            self.assertEqual(events[0]["start_s"],101)
            self.assertEqual(events[0]["end_s"],102)
            for bad in ({"source":""},{"clock":"guess"},{"end_s":20}):
                pd.DataFrame([{**rows[0],**bad}]).to_csv(p,index=False)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    read_annotations(p,VideoClock(1,100,0,1),100,110)

    def test_display_envelope_preserves_peaks_and_gaps(self):
        recording=Recording(pd.DataFrame({"time_s":np.arange(8)/128,"C3":[1,2,50,3,np.nan,1,2,3]}),
                            {"sampling_rate_hz":128,"eeg_columns":["C3"]})
        raw=_raw_display(recording,max_blocks=2)
        self.assertEqual(raw["block_samples"],4)
        self.assertEqual(raw["max_uv"],[ [50.0], [None] ])
        self.assertAlmostEqual(raw["end_s"][0],4/128)


HAS_MEDIA = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@unittest.skipUnless(HAS_MEDIA,"FFmpeg integration tests need ffmpeg and ffprobe on PATH")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.video=Path(cls.temp.name)/"fixture.mp4"
        black=np.zeros((4,48,64,3),np.uint8)
        white=np.full((4,48,64,3),255,np.uint8)
        command=[shutil.which("ffmpeg"),"-y","-v","error","-f","rawvideo","-pixel_format","rgb24",
                 "-video_size","64x48","-framerate","4","-i","pipe:0","-an","-c:v","libx264",
                 "-pix_fmt","yuv420p",str(cls.video)]
        subprocess.run(command,input=np.concatenate([black,white]).tobytes(),capture_output=True,check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fixture(self):
        rng=np.random.default_rng(11)
        frame=pd.DataFrame(rng.normal(size=(128*36,2))*8,columns=["C3","C4"])
        frame.insert(0,"time_s",100+np.arange(len(frame))/128)
        recording=Recording(frame,{"sampling_rate_hz":128,"eeg_unit":"uV","eeg_columns":["C3","C4"],
                            "subject_id":"video-synthetic","source":"synthetic EEG",
                            "recording_context":"synthetic_numerical_demonstration"})
        model={"subject_id":"video-synthetic","medication_profile_id":"not_recorded",
               "loadings":{"C3:log_beta_alpha":1,"C4:log_beta_alpha":1}}
        sync={"subject_id":"video-synthetic","video_kind":"synthetic","source":"synthetic frames",
              "license":"CC0-1.0","evidence":"Known fixture clocks",
              "anchors":[{"video_s":0,"eeg_s":124},{"video_s":1.75,"eeg_s":125.75}],"uncertainty_ms":1}
        return recording,model,sync

    def test_probe_and_frame_difference_detect_known_transition(self):
        info=probe_video(self.video)
        self.assertAlmostEqual(info["duration_s"],2)
        self.assertEqual((info["width"],info["height"]),(64,48))
        motion=frame_difference(self.video,VideoClock(1,100,0,1),fps=4)
        self.assertTrue(np.isnan(motion.frame_difference.iloc[0]))
        np.testing.assert_allclose(motion.frame_difference.iloc[1:4],0,atol=.01)
        self.assertGreater(motion.frame_difference.iloc[4],.8)
        self.assertEqual(motion.video_s.iloc[4],1)
        self.assertEqual(motion.eeg_s.iloc[4],101)

    def test_invalid_motion_configuration_and_container_rejected(self):
        for kw in ({"fps":-1},{"fps":float("nan")},{"roi":[0,0,2,1]}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                frame_difference(self.video,VideoClock(1,100,0,1),**kw)
        with self.assertRaises(ValueError):
            probe_video(Path(self.temp.name)/"missing.mp4")

    def test_report_uses_project_tracker_and_escapes_embedded_metadata(self):
        recording,model,sync=self.fixture()
        recording.metadata["source"]="synthetic </script><script>window.INJECTED=true</script>"
        with tempfile.TemporaryDirectory() as directory:
            summary=write_video_eeg_report(recording,model,24,self.video,sync,directory,draws=64,embed_video=True)
            html=(Path(directory)/"report.html").read_text(encoding="utf-8")
            self.assertIn("SYNTHETIC DEMONSTRATION",html)
            self.assertIn('"base64":',html)
            self.assertNotIn("synthetic </script><script>",html)
            self.assertNotIn("__VIDEO_EEG_PAYLOAD__",html)
            self.assertEqual(summary["monitoring_windows"],11)
            self.assertEqual(summary["video_eeg"]["common_video_interval_s"],[0,2])
            self.assertFalse(summary["video_eeg"]["motion"]["used_in_kalman_measurements"])
            trace=pd.read_csv(Path(directory)/"tracking.csv")
            # A two-second clip ends at EEG 126 s. Later EEG windows cannot
            # inherit motion observations from outside the source video.
            self.assertTrue((trace.loc[trace.window_start_s>=126,"video_motion_samples"]==0).all())
            self.assertIsNone(summary["SOI"])
            self.assertEqual(summary["classification"],"indeterminate")
            json.dumps(summary,allow_nan=False)

    def test_wrong_subject_or_out_of_video_anchor_rejected(self):
        recording,model,sync=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"subject_id"):
                write_video_eeg_report(recording,model,24,self.video,{**sync,"subject_id":"wrong"},directory,draws=64)
            sync["anchors"]=[{"video_s":0,"eeg_s":124},{"video_s":4,"eeg_s":128}]
            with self.assertRaisesRegex(ValueError,"outside the video"):
                write_video_eeg_report(recording,model,24,self.video,sync,directory,draws=64)

    def test_cli_writes_report_and_adjacent_video(self):
        recording,model,sync=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            recording.frame.to_csv(d/"eeg.csv",index=False)
            for name,data in (("metadata",recording.metadata),("model",model),("sync",sync)):
                (d/f"{name}.json").write_text(json.dumps(data),encoding="utf-8")
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=main(["video-eeg",str(d/"eeg.csv"),"--metadata",str(d/"metadata.json"),
                           "--video",str(self.video),"--sync",str(d/"sync.json"),"--model",str(d/"model.json"),
                           "--baseline-sec","24","--bootstrap-draws","64","--motion-fps","0","--out",str(d/"out")])
            self.assertEqual(code,0)
            self.assertTrue((d/"out/report.html").is_file())
            self.assertEqual((d/"out/source_video.mp4").read_bytes(),self.video.read_bytes())
