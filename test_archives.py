import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import archives


class BundleTests(unittest.TestCase):
    def test_roundtrip_and_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source"
            source.mkdir()
            (source / "data.csv").write_text("raw,value\n1,2\n")
            bundle = base / "bundle.tar.gz"
            archives.pack(bundle, source)
            dest = base / "destination"
            dest.mkdir()
            archives.extract(bundle, dest)
            archives.extract(bundle, dest)
            self.assertEqual((dest / "runs/data.csv").read_text(), "raw,value\n1,2\n")
            (dest / "runs/data.csv").write_text("changed")
            with self.assertRaises(ValueError):
                archives.extract(bundle, dest)

    def test_unsafe_members_rejected(self):
        for name, kind in (("../escape", tarfile.REGTYPE), ("runs/link", tarfile.SYMTYPE),
                           ("/runs/absolute", tarfile.REGTYPE), ("runs/../escape", tarfile.REGTYPE)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "bad.tar.gz"
                with tarfile.open(path, "w:gz") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    member.size = 1 if kind == tarfile.REGTYPE else 0
                    member.linkname = "/outside" if kind == tarfile.SYMTYPE else ""
                    archive.addfile(member, io.BytesIO(b"x") if member.size else None)
                with self.assertRaises(ValueError):
                    archives.extract(path, Path(temp) / "destination")


if __name__ == "__main__":
    unittest.main()
