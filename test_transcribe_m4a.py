import unittest

from transcribe_m4a import format_time, label_speakers, merge_segments, render_txt


class TranscriptFormattingTests(unittest.TestCase):
    def test_format_time(self):
        self.assertEqual(format_time(65.432), "01:05.432")
        self.assertEqual(format_time(3661.2), "01:01:01.200")

    def test_merges_adjacent_same_speaker(self):
        segments = [
            {"speaker": "A", "start": 0, "end": 1, "text": "你好"},
            {"speaker": "A", "start": 1, "end": 2, "text": "world"},
            {"speaker": "B", "start": 2, "end": 3, "text": "收到"},
        ]
        merged = merge_segments(segments)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["text"], "你好 world")
        self.assertEqual(merged[0]["end"], 2.0)

    def test_labels_speakers_by_first_appearance(self):
        segments = [
            {"speaker": "SPEAKER_07", "start": 0, "end": 1, "text": "a"},
            {"speaker": "SPEAKER_02", "start": 1, "end": 2, "text": "b"},
            {"speaker": "SPEAKER_07", "start": 2, "end": 3, "text": "c"},
        ]
        labels = [item["speaker"] for item in label_speakers(segments)]
        self.assertEqual(labels, ["讲话人1", "讲话人2", "讲话人1"])

    def test_render_without_timestamps(self):
        text = render_txt(
            [{"speaker": "A", "start": 0, "end": 1, "text": "测试 test"}],
            timestamps=False,
        )
        self.assertEqual(text, "讲话人1: 测试 test\n")


if __name__ == "__main__":
    unittest.main()
