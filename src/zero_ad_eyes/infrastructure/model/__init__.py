"""Model seam adapters (REQUIREMENTS.md §5.10).

``StubPerceptionModel`` (MP3) satisfies the ``PerceptionModel`` port without a
trained model, so the whole pipeline runs today. The real adapter (MP4, loading the
delivered ONNX artifact) is added here later and is the ONLY new code at plug-in
time; parity tests (MP5) assert it satisfies the identical contract.
"""
