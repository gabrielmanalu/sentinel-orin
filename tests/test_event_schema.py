from services.pipeline.event_builder import EVENT_TYPES


def test_five_event_types_defined():
    assert len(EVENT_TYPES) == 5
    assert "cross_camera_handoff" in EVENT_TYPES
    assert "restricted_zone_dwell" in EVENT_TYPES
