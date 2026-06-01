from ignite.engine import Engine, Events
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console_ = Console()


def attach_rich_progress(
    engine: Engine,
    *,
    total: int,
    description: str = "Training",
    console: Console = console_,
):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="blue"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        refresh_per_second=5,
    )

    state = {
        "progress": progress,
        "task_id": None,
    }

    @engine.on(Events.STARTED)
    def _start_progress(engine):
        progress.start()
        state["task_id"] = progress.add_task(description, total=total)

    @engine.on(Events.ITERATION_COMPLETED)
    def _update_progress(engine):
        output = engine.state.output

        # output が loss の float を返す想定
        if isinstance(output, (float, int)):
            progress.update(
                state["task_id"],
                advance=1,
                description=f"{description} | loss={output:.4f}",
            )
        else:
            progress.advance(state["task_id"])

    @engine.on(Events.COMPLETED)
    def _stop_progress(engine):
        progress.stop()


def attach_epoch_rich_progress_per_epoch(engine: Engine, *, num_batches: int):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="blue"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        refresh_per_second=5,
    )

    state = {
        "task_id": None,
    }

    @engine.on(Events.STARTED)
    def _start(engine):
        progress.start()

    @engine.on(Events.EPOCH_STARTED)
    def _epoch_start(engine):
        epoch = engine.state.epoch
        state["task_id"] = progress.add_task(
            f"Epoch {epoch}",
            total=num_batches,
        )

    @engine.on(Events.ITERATION_COMPLETED)
    def _iteration_completed(engine):
        loss = engine.state.output

        if isinstance(loss, (float, int)):
            progress.update(
                state["task_id"],
                advance=1,
                description=f"Epoch {engine.state.epoch} | loss={loss:.4f}",
            )
        else:
            progress.advance(state["task_id"])

    @engine.on(Events.EPOCH_COMPLETED)
    def _epoch_completed(engine):
        progress.remove_task(state["task_id"])

    @engine.on(Events.COMPLETED)
    def _completed(engine):
        progress.stop()
