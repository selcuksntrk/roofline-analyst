"""Click command-line interface for Roofline Analyst."""

from __future__ import annotations

from pathlib import Path

import click

from roofline_analyst.hardware import (
    HardwareBenchmarkError,
    measure_hardware_limits,
)
from roofline_analyst.hooks import capture_module_executions
from roofline_analyst.layer_points import build_layer_roofline_points
from roofline_analyst.models import build_model
from roofline_analyst.profiling import profile_operator_flops
from roofline_analyst.roofline import (
    build_roofline_geometry,
    create_roofline_figure,
)


@click.command()
@click.option(
    "--binary",
    type=click.Path(
        exists=True,
        dir_okay=False,
        executable=True,
        path_type=Path,
    ),
    required=True,
    help="Path to the compiled roofline_hw executable.",
)
@click.option(
    "--model",
    type=click.Choice(["token-mlp"], case_sensitive=False),
    default="token-mlp",
    show_default=True,
    help="Built-in model to profile.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Batch dimension for the example model input.",
)
@click.option(
    "--sequence-length",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="Sequence dimension for the example model input.",
)
@click.option(
    "--output",
    type=click.Path(
        dir_okay=False,
        writable=True,
        path_type=Path,
    ),
    default=Path("roofline.html"),
    show_default=True,
    help="Output HTML path for the interactive chart.",
)
def main(
        binary: Path,
        model: str,
        batch_size: int,
        sequence_length: int,
        output: Path,
) -> None:
    """Measure CPU limits and render a built-in model roofline chart."""
    if not output.parent.exists():
        raise click.ClickException(
            f"output directory does not exist: {output.parent}"
        )

    try:
        hardware_limits = measure_hardware_limits(binary)
        model_instance, example_inputs = build_model(
            model,
            batch_size,
            sequence_length,
        )

        operator_profiles = profile_operator_flops(
            model_instance,
            example_inputs,
        )
        executions = capture_module_executions(
            model_instance,
            example_inputs,
        )
        layer_points = build_layer_roofline_points(
            model_instance,
            executions,
            hardware_limits,
        )
        geometry = build_roofline_geometry(hardware_limits)
        figure = create_roofline_figure(geometry, layer_points)
        figure.write_html(output, include_plotlyjs=True)
    except (HardwareBenchmarkError, ValueError) as error:
        raise click.ClickException(str(error)) from error

    operator_flops = sum(profile.flops for profile in operator_profiles)

    click.echo(f"Wrote roofline chart: {output}")
    click.echo(f"Operator-level profiler FLOPs: {operator_flops}")
    click.echo(f"Supported roofline points: {len(layer_points)}")