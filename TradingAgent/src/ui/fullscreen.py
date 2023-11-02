"""
Demonstrates a Rich "application" using the Layout and Live classes.

"""

from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

console = Console()
console.log("Server starting...")

def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=7),
    )
    layout["main"].split_row(
        Layout(name="side"),
        Layout(name="body", ratio=2, minimum_size=60),
    )
    layout["side"].split(Layout(name="box1"), Layout(name="box2"))
    return layout


def make_sponsor_message() -> Panel:
    """Some example content."""
    sponsor_message = Table.grid(padding=1)
    sponsor_message.add_column(style="green", justify="right")
    sponsor_message.add_column(no_wrap=True)
    sponsor_message.add_row(
        "Telegram:",
        "[u blue link=https://t.me/xsa_tg]you can chat with me in private",
    )
    sponsor_message.add_row(
        "Website:",
        "[u blue link=https://alekseysavin.com]made by alekseysavin.com",
    )
    sponsor_message.add_row(
        "Microblog:", "[u blue link=https://t.me/xsa_logs]subscribe to my (b)log"
    )

    message = Table.grid(padding=1)
    message.add_column()
    message.add_column(no_wrap=True)
    message.add_row(sponsor_message)

    message_panel = Panel(
        Align.center(
            Group("\n", Align.center(sponsor_message)),
            vertical="middle",
        ),
        box=box.ROUNDED,
        padding=(1, 2),
        title="[b red]Thanks for trying!",
        border_style="bright_blue",
    )
    return message_panel


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            "[b]ALXY:[/b]Trading Bot",
            datetime.now().ctime().replace(":", "[blink]:[/]"),
        )
        return Panel(grid, style="white on blue")


def make_syntax() -> Syntax:
    code = """\
    if sell > sell, sell = sell
    if buy > buy, buy = buy

    profit = sell - buy

    reward = profit++
    ...
    """
    syntax = Syntax(code, "python", line_numbers=True)
    return syntax

def make_table() -> Table:
    table = Table.grid()
    table = Table(title="Your open positions")

    table.add_column("Open at", style="cyan", no_wrap=True)
    table.add_column("Open price", style="magenta")
    table.add_column("Side", justify="right", style="green")
    table.add_column("Take profit", justify="right", style="green")
    table.add_column("Stop loss", justify="right", style="red")

    table.add_row("Dec 20, 2019 12:12", "23554",  "BTC/USDT", "0.01", "-0.025", "$952,110,690")
    table.add_row("Dec 20, 2018 12:33", "23555", "BTC/USDT", "0.01", "-0.025", "$393,151,347")
    table.add_row("Dec 20, 2017 14:22", "22112", "BTC/USDT", "0.01", "-0.025", "$1,332,539,889")
    table.add_row("Dec 20, 2016 14:35", "22444", "BTC/USDT" "0.01", "-0.025", "$1,332,439,889")
    table.add_row("Dec 20, 2019 12:12", "23554",  "BTC/USDT", "0.01", "-0.025", "$952,110,690")
    table.add_row("Dec 20, 2018 12:33", "23555", "BTC/USDT", "0.01", "-0.025", "$393,151,347")
    table.add_row("Dec 20, 2017 14:22", "22112", "BTC/USDT", "0.01", "-0.025", "$1,332,539,889")
    table.add_row("Dec 20, 2016 14:35", "22444", "BTC/USDT" "0.01", "-0.025", "$1,332,439,889")

    return table


def demo():
    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    job_progress.add_task("[green]Training")
    job_progress.add_task("[magenta]Evaluating", total=200)
    job_progress.add_task("[cyan]Traiding", total=400)

    total = sum(task.total for task in job_progress.tasks)
    overall_progress = Progress()
    overall_task = overall_progress.add_task("All Jobs", total=int(total))

    progress_table = Table.grid(expand=True)
    progress_table.add_row(
        Panel(
            overall_progress,
            title="Overall Progress",
            border_style="green",
            padding=(2, 2),
        ),
        Panel(job_progress, title="[b]Jobs", border_style="red", padding=(1, 2)),
    )

    layout = make_layout()
    layout["header"].update(Header())
    layout["body"].update(make_table())
    layout["box2"].update(Panel(make_syntax(), border_style="green"))
    layout["box1"].update(Panel(make_sponsor_message(), border_style="red"))
    layout["footer"].update(progress_table)

    from time import sleep
    from rich.live import Live

    with Live(layout, refresh_per_second=10, screen=True):
        while not overall_progress.finished:
            sleep(0.1)
            for job in job_progress.tasks:
                if not job.finished:
                    job_progress.advance(job.id)

            completed = sum(task.completed for task in job_progress.tasks)
            overall_progress.update(overall_task, completed=completed)
