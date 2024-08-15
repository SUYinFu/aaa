import flet as ft
class IconRectButton(ft.Container):
    def __init__(
        self, icon, icon_size=None, width=None, height=None, elevation=0, on_click=None
    ):
        super().__init__()
        self._icon_btn = ft.IconButton(
            icon=icon,
            icon_size=icon_size,
            width=width,
            height=height,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=5),
            ),
            on_click=on_click,
        )
        self.__card = ft.Card(
            elevation=elevation,
            shape=ft.RoundedRectangleBorder(radius=5),
            content=ft.Container(
                content=ft.Column(
                    spacing=0,
                    horizontal_alignment="center",
                    controls=[
                        self._icon_btn,
                    ],
                )
            ),
        )
        self.content = self.__card


class BarButton(ft.Container):
    def __init__(self, text_btn="close", page=None):
        super().__init__()
        if text_btn == "close":
            text = "✕"
            on_hover = self.on_hover_close
            on_click = page.window.close
        elif text_btn == "min":
            text = "一"
            on_hover = self.on_hover_min
            on_click = self.window_minimize
        elif text_btn == "max":
            text = "☐"
            on_hover = self.on_hover_max
            on_click = self.window_maximize
        self.content = ft.ElevatedButton(
            text=text,
            elevation=0,
            height=35,
            on_click=lambda _: on_click(),
            on_hover=on_hover,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=0)),
            bgcolor=ft.colors.BACKGROUND,
        )

    def window_minimize(self):
        self.page.window.minimized = True
        self.page.update()

    def window_maximize(self):
        if self.page.window.maximized == True:
            self.page.window.maximized = False
            self.content.text = "☐"
        else:
            self.page.window.maximized = True
            self.content.text = "❐"
        self.page.update()

    def on_hover_min(self, e):
        e.control.bgcolor = (
            ft.colors.BLUE_400 if e.data == "true" else ft.colors.BACKGROUND
        )
        e.control.update()

    def on_hover_max(self, e):
        e.control.bgcolor = (
            ft.colors.BLUE_400 if e.data == "true" else ft.colors.BACKGROUND
        )
        e.control.update()

    def on_hover_close(self, e):
        e.control.bgcolor = (
            ft.colors.RED_400 if e.data == "true" else ft.colors.BACKGROUND
        )
        e.control.update()


def NewBar(page):
    return ft.AppBar(
        leading=ft.WindowDragArea(
            ft.Container(
                ft.Icon(
                    ft.icons.ALL_INCLUSIVE_OUTLINED,
                    color=ft.colors.CYAN_900,
                    scale=1.25,
                ),
                margin=ft.margin.only(left=10),
            ),
        ),
        title=ft.WindowDragArea(
            ft.Container(
                ft.Row(
                    [
                        ft.Text("All-Tools", size=14, color=ft.colors.BLUE_GREY),
                    ],
                ),
                margin=ft.margin.only(left=30),
            ),
        ),
        leading_width=0,
        toolbar_height=35,
        center_title=False,
        color=ft.colors.BACKGROUND,
        bgcolor=ft.colors.BACKGROUND,
        actions=[BarButton("min"), BarButton("max"), BarButton("close", page)],
    )