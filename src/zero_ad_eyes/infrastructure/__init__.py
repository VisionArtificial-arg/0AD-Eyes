"""Infrastructure: concrete adapters that implement the application ports.

This is the outer onion ring — the only place OpenCV, screen-capture libraries,
and (later) the model runtime are imported. Feature agents add packages here
(``acquisition``, ``preprocessing``, ``calibration``, ``perception``, ``model`` …).
"""
