"""
face_id — lightweight few-shot face verification library.

Drop this package into any project and call two functions:

    from face_id import enroll, verify

    enroll("alice", "alice_photo.jpg")
    result = verify("camera_frame.jpg")
    print(result)  # {"match": "alice", "confidence": 0.87, "verified": True}
"""

from face_id.pipeline import (
    enroll,
    verify,
    enroll_from_array,
    verify_from_array,
    verify_with_details,
)

__all__ = [
    "enroll",
    "verify",
    "enroll_from_array",
    "verify_from_array",
    "verify_with_details",
]
__version__ = "0.1.0"
