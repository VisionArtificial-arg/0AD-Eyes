"""Application layer: ports (the seams) and pipeline orchestration.

Ports are ``typing.Protocol`` interfaces expressed purely in domain terms. Every
adapter in ``infrastructure`` implements one of them; the ``PerceptionPipeline``
wires them together without knowing which concrete implementation it holds.
"""
