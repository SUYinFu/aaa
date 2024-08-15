import flet as ft
from NvBar.TopBar import NewBar
from charts.linecharts import RealTimeChart

# 导入MetaTrader5库，用于获取市场数据
import MetaTrader5 as mt5

# 初始化MetaTrader5，如果失败则打印错误信息并退出
if not mt5.initialize(login=76666059, server="Exness-MT5Trial5", password="@Kassd2012"):
    print("initialize() failed, error code =", mt5.last_error())
    quit()





class State:
    toggle = True


s = State()


def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.spacing = 0
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.window.always_on_top = True
    page.window.title_bar_hidden = True
    page.window.title_bar_buttons_hidden = True
    page.window.width, page.window.height = 350, 600
    # page.window.max_width, page.window.max_height = 350, 600
    # page.window.resizable = False

    page.appbar = NewBar(page)
    rail = ft.NavigationRail(
        selected_index=0,
        indicator_shape=ft.ContinuousRectangleBorder(),
        indicator_color=ft.colors.TRANSPARENT,
        label_type=ft.NavigationRailLabelType.NONE,
        min_width=25,
        min_extended_width=150,
        leading=ft.FloatingActionButton(
            icon=ft.icons.MENU,
            bgcolor=ft.colors.TRANSPARENT,
            on_click=lambda e: open_rail(e),
            mini=True,
        ),
        group_alignment=-1.0,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.icons.HOME_OUTLINED,
                selected_icon=ft.icons.HOME,
                label_content=ft.Text("Home", color=ft.colors.GREY),
                padding=ft.padding.only(left=10, right=10, bottom=1),
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.CANDLESTICK_CHART_OUTLINED,
                selected_icon=ft.icons.CANDLESTICK_CHART_ROUNDED,
                label_content=ft.Text("Kline", color=ft.colors.GREY),
                padding=ft.padding.only(left=10, right=10, bottom=1),
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.ACCOUNT_BALANCE_WALLET,
                selected_icon=ft.icons.ACCOUNT_BALANCE_WALLET_OUTLINED,
                label_content=ft.Text("Account", color=ft.colors.GREY),
                padding=ft.padding.only(left=10, right=10, bottom=1),
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.SETTINGS_OUTLINED,
                selected_icon=ft.icons.SETTINGS,
                label_content=ft.Text("Settings", color=ft.colors.GREY),
                padding=ft.padding.only(left=10, right=10, bottom=1),
            ),
        ],
        on_change=lambda e: change_rail(e),
    )

    def change_rail(e):
        i = e.control.selected_index
        print("Selected destination:", e.control.selected_index)

        # e.control.destinations[i].label_content.color=ft.colors.WHITE

        # e.control.update()

    def open_rail(e):
        if rail.extended == False:
            rail.extended = True
        else:
            rail.extended = False
        rail.update()

    data_1 = [
        ft.LineChartData(
            data_points=[
                ft.LineChartDataPoint(0, 3),
                ft.LineChartDataPoint(2.6, 2),
                ft.LineChartDataPoint(4.9, 5),
                ft.LineChartDataPoint(6.8, 3.1),
                ft.LineChartDataPoint(8, 4),
                ft.LineChartDataPoint(9.5, 3),
                ft.LineChartDataPoint(11, 6),
            ],
            stroke_width=2,
            color=ft.colors.CYAN,
            curved=True,
            stroke_cap_round=True,
            selected_below_line=False,
        )
    ]
    
    # 获取XAUUSDm（黄金）和BTCUSDm（比特币）的市场数据，时间间隔为5分钟
    rates = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_M5, 0, 100)
    print(rates)
    # 将数据转换为[(x, y)]格式，其中x是时间点，y是价格
    POINTS = [(i + 1, rate[1]) for i, rate in enumerate(rates)]
    gold_chart = RealTimeChart(data=POINTS, label="Gold", color=ft.colors.GREEN)
    page.add(
        ft.Row(
            [
                ft.Card(rail, elevation=15, margin=ft.margin.only(right=1)),
                # ft.VerticalDivider(width=1),
                ft.Card(
                    ft.Container(
                        ft.Column(
                            expand=True,
                            alignment="center",
                            horizontal_alignment="center",
                            controls=[
                                ft.Container(
                                    expand=4,
                                    content=gold_chart,
                                    padding=20,
                                    border_radius=6,
                                    bgcolor=ft.colors.with_opacity(
                                        0.005, ft.colors.WHITE10
                                    ),
                                ),
                            ],
                        ),
                        # padding=5,
                    ),
                    elevation=1,
                    margin=ft.margin.only(right=1),
                    expand=1,
                    shape=ft.ContinuousRectangleBorder(),
                ),
            ],
            expand=True,
            spacing=0,
        )
    )


ft.app(target=main)
