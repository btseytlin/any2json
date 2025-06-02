from any2json.containers import Sample


class SampleGenerator:
    def generate_samples(
        self,
        source_data: dict | list | None = None,
        source_schema: dict | None = None,
        *args,
        **kwargs,
    ) -> list[Sample] | None:
        raise NotImplementedError
