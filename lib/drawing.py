from typing import Optional

from IPython.display import display
import ipyturtle3 as turtle


def move_turtle(t: turtle.Turtle, label: bool, step_size_up: int, step_size_right: int):
    if label:
        t.setheading(90)
        t.forward(step_size_up)
    elif not label:
        t.setheading(0)
        t.forward(step_size_right)


def set_up_turtle_screen(canvas_width=500, canvas_height=500, margin=5) -> turtle.TurtleScreen:
    cv = turtle.Canvas(width=canvas_width + 2 * margin, height=canvas_height + 2 * margin)
    display(cv)
    screen = turtle.TurtleScreen(cv)
    pg = [
        -canvas_width // 2,
        canvas_height // 2,
        canvas_width // 2,
        canvas_height // 2,
        canvas_width // 2,
        -canvas_height // 2,
        -canvas_width // 2,
        -canvas_height // 2,
    ]
    cv.create_polygon(pg, fill="white", outline="black")
    cv.create_line(
        -canvas_width // 2,
        canvas_height // 2,
        canvas_width // 2,
        -canvas_height // 2,
        fill="grey",
        width=1,
    )
    return screen


def get_sorted_labels(y_true: list[bool], y_pred: list[float]) -> list[bool]:
    labels_scores = list(zip(y_true, y_pred))
    labels_scores.sort(key=lambda x: x[1], reverse=True)
    labels_sorted = [x[0] for x in labels_scores]
    return labels_sorted


def start_turtle_at_the_bottom_left(t: turtle.Turtle, canvas_width=500, canvas_height=500):
    start_x, start_y = -canvas_width // 2, -canvas_height // 2
    t.teleport(start_x, start_y)
    t.setheading(0)


def draw_roc_curve_with_a_turtle(
    y_true: list[bool],
    y_pred: list[float],
    color: str = "blue",
    stop_at: Optional[float] = None,
) -> turtle.Canvas:
    return draw_roc_curve_with_multiple_turtles(y_true, [y_pred], [color], stop_at=stop_at)


def draw_roc_curve_with_multiple_turtles(
    y_true: list[bool],
    y_pred: list[list[float]],
    colors: list[str],
    canvas_width=500,
    canvas_height=500,
    margin=20,
    stop_at: Optional[float] = None,
) -> turtle.Canvas:
    screen = set_up_turtle_screen(canvas_width, canvas_height, margin)

    number_of_positive_labels = sum(y_true)
    number_of_negative_labels = len(y_true) - sum(y_true)

    step_size_up = canvas_height // number_of_positive_labels
    step_size_right = canvas_width // number_of_negative_labels

    turtles = [turtle.Turtle(screen, shape="turtle") for _ in range(len(y_pred))]

    for t, color in zip(turtles, colors):
        t.color(color)
        start_turtle_at_the_bottom_left(t, canvas_width, canvas_height)

    labels_sorted = []
    for y_pred_per_classifier in y_pred:
        labels_sorted.append(get_sorted_labels(y_true, y_pred_per_classifier))

    if stop_at:
        path_range = int(stop_at * len(y_true))
    else:
        path_range = len(y_true)

    with turtle.hold_canvas(screen.cv):
        for i in range(path_range):
            for j, t in enumerate(turtles):
                move_turtle(t, labels_sorted[j][i], step_size_up, step_size_right)

    return screen.cv
