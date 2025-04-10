
# for flexible attention calling paired with auto classes.
class AttentionConfig:
    def __init__(
        self,
        attention_type: str = "full",
        num_heads: int = 4,
        dropout: float = 0.1,
        kwargs: dict = None,
    ):
        assert isinstance(dropout, (float, int)), f"`dropout` must be float or int, got {type(dropout)}"
        assert 0.0 <= dropout <= 1.0, f"`dropout` must be between 0 and 1, got {dropout}"

        self.attention_type = attention_type
        self.num_heads = num_heads
        self.dropout = float(dropout)  # cast to float for consistency
        self.kwargs = kwargs or {}

    def resolve_class(self):
        return ATTENTION_REGISTRY[self.attention_type]

    def build(self):
        attn_cls = self.resolve_class()
        return attn_cls(
            num_heads=self.num_heads,
            dropout=self.dropout,
            **self.kwargs
        )
