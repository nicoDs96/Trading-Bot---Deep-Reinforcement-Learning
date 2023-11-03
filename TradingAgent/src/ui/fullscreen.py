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
from rich.table import Table


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            "[white on blue]Trading Bot, ",
            datetime.now().ctime().replace(":", "[blink]:[/]"),
        )
        return Panel(Align.center(grid, style="white on blue"))


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
        Layout(name="body", minimum_size=60),
    )
    layout["side"].split(Layout(name="box1"), Layout(name="box2"))
    layout["body"].split(Layout(name="body1"), Layout(name="body2"))
    return layout


def make_sponsor_message() -> Panel:
    """Some example content."""
    sponsor_message = Table.grid()
    sponsor_message.add_column(style="green")
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

    message = Table.grid(padding=0)
    message.add_row(sponsor_message)

    message_panel = Panel(
        Align.center(
            Group("\n", Align.center(sponsor_message)),
            vertical="top",
        ),
    )
    return sponsor_message


def make_trade_vector() -> Panel:
    """Some example content."""
    sponsor_message = Table.grid(padding=1)
    sponsor_message.add_column(style="green", justify="right")
    sponsor_message.add_column(no_wrap=True)

    from rich.text import Text
    from rich.spinner import Spinner, SPINNERS

    spinner_name = "noise"
    import random as rnd

    colors = ["red", "green", "white"]
    coins = ["DOGE", "SHIB", "SOL","BTC", "USDT" ]

    for coin in coins[::-1]:
        sponsor_message.add_row(
            coin,
            *[Spinner(spinner_name, style=rnd.choice(colors)) for i in range(1, 30)],
        )

    return sponsor_message


from rich.console import Console
from rich.panel import Panel


def make_logs_reader() -> Panel:
    console = Console(record=True)
    # Print to console
    messages = [
        "Last server tick 2023-11-03 12:03:00 with `Close` price 34195.01",
        "Loaded 1000 BTC/USDT candles in total",
        "Loaded ohlcv: (1000, 6)",
        "Last server tick 2023-11-03 12:04:00 with `Close` price 34183.92",
        "Updated data for last tick: 1699013040000 last price: 34183.92",
        ".........",
        "Profit:  10",
        "Value: 20",
        "exit state:",
        "(2, -5.0, False, -1, [])",
        "signal sent to dashboards..",
    ]
    for message in messages:
        console.print(message)

    # Create a panel with the console output
    console_output = console.export_text()
    panel = Panel(console_output)

    return panel


def make_table(side="Sell") -> Table:
    table = Table.grid()
    table = Table(title=f"Your open {side} positions")

    if side == "Sell":
        table.add_column("Time", style="cyan", no_wrap=True)
        table.add_column("Price", style="magenta")
        table.add_column("Side", justify="right", style="red")
        table.add_column("Take", justify="right", style="red")
        table.add_column("Loss", justify="right", style="red")

        table.add_column("Value", justify="right", style="blue")

        table.add_row(
            "Dec 20, 2022 11:12", "23554", side, "0.01", "-0.025", "$952,110,690"
        )
        table.add_row(
            "Dec 20, 2022 12:33", "23555", side, "0.01", "-0.025", "$393,151,347"
        )
        table.add_row(
            "Dec 20, 2022 13:22", "22112", side, "0.01", "-0.025", "$1,332,539,889"
        )
        table.add_row(
            "Dec 20, 2022 14:12", "22444", side, "0.01", "-0.025", "$1,332,439,889"
        )
        table.add_row(
            "Dec 20, 2022 14:22", "23554", side, "0.01", "-0.025", "$952,110,690"
        )
        table.add_row(
            "Dec 20, 2022 15:33", "23555", side, "0.01", "-0.025", "$393,151,347"
        )
        table.add_row(
            "Dec 20, 2022 16:22", "22112", side, "0.01", "-0.025", "$1,332,539,889"
        )
        table.add_row(
            "Dec 20, 2022 17:35", "22444", side, "0.01", "-0.025", "$1,332,439,889"
        )

    if side == "Buy":
        table.add_column("Time", style="cyan", no_wrap=True)
        table.add_column("Price", style="magenta")
        table.add_column("Side", justify="right", style="green")
        table.add_column("Take", justify="right", style="green")
        table.add_column("Loss", justify="right", style="green")
        table.add_column("Value", justify="right", style="blue")

        table.add_row(
            "Dec 20, 2022 12:12", "23554", side, "0.01", "-0.025", "$952,110,690"
        )
        table.add_row(
            "Dec 20, 2022 12:33", "23555", side, "0.01", "-0.025", "$393,151,347"
        )
        table.add_row(
            "Dec 20, 2022 14:22", "22112", side, "0.01", "-0.025", "$1,332,539,889"
        )
        table.add_row(
            "Dec 20, 2022 14:35", "22444", side, "0.01", "-0.025", "$1,332,439,889"
        )
        table.add_row(
            "Dec 20, 2022 12:12", "23554", side, "0.01", "-0.025", "$952,110,690"
        )
        table.add_row(
            "Dec 20, 2022 12:33", "23555", side, "0.01", "-0.025", "$393,151,347"
        )
        table.add_row(
            "Dec 20, 2022 14:22", "22112", side, "0.01", "-0.025", "$1,332,539,889"
        )
        table.add_row(
            "Dec 20, 2022 14:35", "22444", side, "0.01", "-0.025", "$1,332,439,889"
        )

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
            border_style="blue",
            padding=(2, 2),
        ),
        Panel(job_progress, title="[b]Jobs", border_style="blue", padding=(1, 2)),
    )

    layout = make_layout()
    layout["header"].update(Header())
    layout["body1"].update(
        Group(
            Align.center(Panel(make_sponsor_message(), border_style="white")),
            Panel(make_trade_vector(), border_style="white"),
        )
    )
    layout["body2"].update(
        Align.center(Panel(make_table(side="Sell"), border_style="red"))
    )
    layout["box2"].update(
        Align.center(Panel(make_table(side="Buy"), border_style="green"))
    )
    layout["box1"].update(Align.center(Panel(make_logs_reader(), border_style="white")))
    layout["footer"].update(Align.center((progress_table)))

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
