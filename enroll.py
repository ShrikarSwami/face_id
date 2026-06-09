"""
enroll.py — CLI tool to enroll one or more reference images for an identity.

Usage:
    python enroll.py alice photos/alice1.jpg
    python enroll.py alice photos/alice1.jpg photos/alice2.jpg
    python enroll.py --list
    python enroll.py --remove alice
"""

import argparse
import sys
from pathlib import Path

from face_id.pipeline import enroll
from face_id.gallery_manager import get_gallery


def main():
    parser = argparse.ArgumentParser(
        description="Enroll a face identity into the face_id gallery."
    )
    parser.add_argument("name", nargs="?", help="Identity name (e.g. 'alice')")
    parser.add_argument("images", nargs="*", help="One or more image file paths")
    parser.add_argument("--list", action="store_true", help="List enrolled identities")
    parser.add_argument("--remove", metavar="NAME", help="Remove an enrolled identity")

    args = parser.parse_args()

    gallery = get_gallery()

    if args.list:
        identities = gallery.list_identities()
        if not identities:
            print("Gallery is empty. No identities enrolled yet.")
        else:
            print(f"Enrolled identities ({len(identities)}):")
            for name in identities:
                print(f"  {name}  ({gallery.count(name)} reference image(s))")
        return

    if args.remove:
        removed = gallery.remove(args.remove)
        if removed:
            print(f"Removed '{args.remove}' from gallery.")
        else:
            print(f"Identity '{args.remove}' not found in gallery.")
        return

    if not args.name or not args.images:
        parser.print_help()
        sys.exit(1)

    for image_path in args.images:
        result = enroll(args.name, image_path)
        if result["enrolled"]:
            print(
                f"✓  Enrolled '{result['name']}' "
                f"({result['count']} reference image(s) total)"
            )
        else:
            print(f"✗  Failed to enroll from '{image_path}': {result['error']}")


if __name__ == "__main__":
    main()
