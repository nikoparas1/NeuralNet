import numpy as np
from PIL import Image
from ipywidgets import IntSlider, link, HBox, VBox, Button, Layout, Output
from ipycanvas import MultiCanvas, hold_canvas


class DrawingWidget(object):
    drawing = False
    position = None
    shape = []
    output_array = None
    drawing_line_width = 3
    history = []
    future = []
    max_history = 100
    alpha = 1.0
    default_style = "#FFFFFF"
    default_radius = 10
    background = "#000000"
    container_bg = "#111111"

    def __init__(
        self,
        width,
        height,
        background="#000000",
        alpha=1.0,
        default_style="#FFFFFF",
        default_radius=10,
    ):
        self.background = background
        self.alpha = alpha
        self.default_style = default_style
        self.default_radius = default_radius
        self.init_canvas(width, height)

    def get_image_data(
        self, background=False, mnist=False, size=28, dtype="float32", invert=False
    ):
        arr = (
            self.canvas.get_image_data()
            if background
            else self.canvas._canvases[1].get_image_data()
        )
        if not mnist:
            return arr

        a = arr[..., 3].astype(np.float32) / 255.0
        if invert:
            a = 1.0 - a

        if size is not None and (a.shape[0] != size or a.shape[1] != size):
            a_resized_u8 = (a * 255.0).astype(np.uint8)
            a = (
                np.array(
                    Image.fromarray(a_resized_u8).resize((size, size), Image.BILINEAR)
                ).astype(np.float32)
                / 255.0
            )

        if dtype == "float32":
            img = a.astype(np.float32)
        elif dtype == "uint8":
            img = (a * 255.0).astype(np.uint8)
        else:
            raise ValueError("dtype must be 'float32' or 'uint8'")

        return img

    def init_canvas(self, width, height):
        self.canvas = MultiCanvas(
            n_canvases=3, width=width, height=height, sync_image_data=True
        )
        self.canvas._canvases[1].sync_image_data = True
        self.reset_background()
        self.canvas.on_mouse_down(self.on_mouse_down)
        self.canvas.on_mouse_move(self.on_mouse_move)
        self.canvas.on_mouse_up(self.on_mouse_up)
        self.canvas[2].stroke_style = "#FFFFFF"
        self.canvas[2].fill_style = "#FFFFFF"
        self.canvas[2].line_cap = "round"
        self.canvas[2].line_width = self.drawing_line_width
        self.canvas[1].stroke_style = self.default_style
        self.canvas[1].line_cap = "round"
        self.canvas[1].line_join = "round"
        self.canvas[1].line_width = self.default_radius
        self.canvas[1].global_alpha = self.alpha

    def reset_background(self, *args):
        with hold_canvas():
            if type(self.background) is np.ndarray:
                self.canvas[0].put_image_data(self.background)
            else:
                self.canvas[0].fill_style = self.background
                self.canvas[0].fill_rect(0, 0, self.canvas.width, self.canvas.height)

    def show(self, on_predict=None):
        radius_slider = IntSlider(
            description="Brush radius:", value=self.default_radius, min=1, max=100
        )
        clear_button = Button(description="Clear")
        clear_button.on_click(self.clear_canvas)
        undo_button = Button(description="Undo", icon="rotate-left")
        undo_button.on_click(self.undo)
        redo_button = Button(description="Redo", icon="rotate-right")
        redo_button.on_click(self.redo)
        predict_button = Button(description="Predict")
        self._pred_out = Output()

        def _do_predict(_):
            if on_predict is None:
                return
            img = self.get_image_data(mnist=True, size=28, dtype="float32")
            with self._pred_out:
                from IPython.display import clear_output

                clear_output()
                on_predict(img)

        predict_button.on_click(_do_predict)

        link((radius_slider, "value"), (self.canvas[1], "line_width"))
        controls = VBox(
            (
                radius_slider,
                clear_button,
                HBox((undo_button, redo_button)),
                predict_button,
                self._pred_out,
            )
        )
        return HBox(
            (self.canvas, controls),
            layout=Layout(background_color=self.container_bg, padding="8px"),
        )

    def save_to_history(self):
        self.history.append(self.canvas._canvases[1].get_image_data())
        if len(self.history) > self.max_history:
            self.history = self.history[1:]
        self.future = []

    def on_mouse_down(self, x, y):
        self.drawing = True
        self.position = (x, y)
        self.shape = [self.position]
        self.save_to_history()

    def on_mouse_move(self, x, y):
        if not self.drawing:
            return
        with hold_canvas():
            self.canvas[2].line_width = self.canvas[1].line_width
            self.canvas[2].stroke_line(self.position[0], self.position[1], x, y)
            self.canvas[2].line_width = self.drawing_line_width
            self.position = (x, y)
        self.shape.append(self.position)

    def on_mouse_up(self, x, y):
        self.drawing = False
        with hold_canvas():
            self.canvas[2].clear()
            self.canvas[1].stroke_lines(self.shape)
        self.shape = []

    def clear_canvas(self, *args):
        self.save_to_history()
        with hold_canvas():
            self.canvas[1].clear()

    def undo(self, *args):
        if self.history:
            with hold_canvas():
                self.future.append(self.canvas._canvases[1].get_image_data())
                self.canvas[1].clear()
                self.canvas[1].put_image_data(self.history[-1])
                self.history = self.history[:-1]

    def redo(self, *args):
        if self.future:
            with hold_canvas():
                self.history.append(self.canvas._canvases[1].get_image_data())
                self.canvas[1].clear()
                self.canvas[1].put_image_data(self.future[-1])
                self.future = self.future[:-1]
