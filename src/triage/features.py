"""Feature contract shared by training and the API."""

# Order matters: the model is trained on arrays that follow this order.
FEATURES: list[str] = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
