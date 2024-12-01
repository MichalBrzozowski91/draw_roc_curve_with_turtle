from IPython.display import display
import ipyturtle3 as turtle

def move_turtle(t: turtle.Turtle, label: bool, step_size_up: int, step_size_right: int):
    if label:
        t.setheading(90)
        t.forward(step_size_up)
    elif not label:
        t.setheading(0)
        t.forward(step_size_right)


def set_up_turtle_screen(canvas_width=500, canvas_height=500, margin=20) -> turtle.TurtleScreen:
    canvas_width += margin
    canvas_height += margin
    cv = turtle.Canvas(width=canvas_width,height=canvas_height)
    display(cv)
    screen = turtle.TurtleScreen(cv)
    start_x, start_y = -screen.canvheight//2, -screen.canvwidth//2
    cv.create_line(start_x, start_y, start_x + canvas_width, start_y, fill='black', width=1)
    cv.create_line(start_x, start_y, start_x, start_y + canvas_height, fill='black', width=1)
    cv.create_line(start_x + canvas_width, start_y, start_x + canvas_width, start_y + canvas_height, fill='black', width=1)
    cv.create_line(start_x, start_y + canvas_height, start_x + canvas_width, start_y + canvas_height, fill='black', width=1)
    cv.create_line(start_x, start_y + canvas_height, start_x + canvas_width, start_y, fill='grey', width=1)
    return screen


def draw_roc_curve_with_a_turtle(y_true: list[bool], y_pred: list[float], color):
    labels_scores = list(zip(y_true, y_pred))
    labels_scores.sort(key=lambda x: x[1], reverse=True)
    labels_sorted = [x[0] for x in labels_scores]

    screen = set_up_turtle_screen()

    number_of_positive_labels = sum(y_true)
    number_of_negative_labels = len(y_true) - sum(y_true)

    step_size_up = screen.canvheight // number_of_positive_labels
    step_size_right = screen.canvheight // number_of_negative_labels

    t = turtle.Turtle(screen)
    t.shape("turtle")
    t.color(color)

    start_x, start_y = -screen.canvheight//2, -screen.canvwidth//2
    t.teleport(start_x, start_y)
    t.setheading(0)
    for label in labels_sorted:
        move_turtle(t, label, step_size_up, step_size_right)


def get_sorted_labels(y_true: list[bool], y_pred: list[float]) -> list[bool]:
    labels_scores = list(zip(y_true, y_pred))
    labels_scores.sort(key=lambda x: x[1], reverse=True)
    labels_sorted = [x[0] for x in labels_scores]
    return labels_sorted


def draw_roc_curve_with_multiple_turtles(y_true: list[bool], y_pred: list[list[float]], colors: list[str]):
    screen = set_up_turtle_screen()

    number_of_positive_labels = sum(y_true)
    number_of_negative_labels = len(y_true) - sum(y_true)

    step_size_up = screen.canvheight // number_of_positive_labels
    step_size_right = screen.canvheight // number_of_negative_labels

    start_x, start_y = -screen.canvheight//2, -screen.canvwidth//2
    turtles = [turtle.Turtle(screen, shape="turtle") for _ in range(len(y_pred))]
    for t, color in zip(turtles, colors):
        t.color(color)
        t.teleport(start_x, start_y)
        t.setheading(0)

    labels_sorted = []
    for y_pred_per_classifier in y_pred:
        labels_sorted.append(get_sorted_labels(y_true, y_pred_per_classifier))

    for i in range(len(y_true)):
        for j, t in enumerate(turtles):
            move_turtle(t, labels_sorted[j][i], step_size_up, step_size_right)
