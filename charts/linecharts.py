import flet as ft
# 定义一个实时图表的类，继承自Flet的Container容器
class RealTimeChart(ft.Container):
    def __init__(self, data, label, color):
        super().__init__()
        self.data = data  # 数据点
        self.color = color  # 线条颜色
        self.label = label  # 图表标签
        self.chart = self.build_chart()  # 创建图表
        self.build()  # 构建UI组件

    # 构建图表的方法
    def build_chart(self):
        # 获取Y轴和X轴的最小值和最大值
        min_y = int(min(self.data, key=lambda y: y[1])[1])
        max_y = int(max(self.data, key=lambda y: y[1])[1])
        min_x = int(min(self.data, key=lambda x: x[0])[0])
        max_x = int(max(self.data, key=lambda x: x[0])[0])

        # 调整Y轴标签显示的间隔，使其刻度更加合理
        y_interval = (max_y - min_y) // 5
        # 调整X轴标签显示的间隔
        x_interval = max_x // 10

        # 创建折线图组件
        chart = ft.LineChart(
            tooltip_bgcolor=ft.colors.with_opacity(0.8, ft.colors.WHITE),
            min_y=min_y,
            max_y=max_y,
            min_x=min_x,
            max_x=max_x,
            expand=True,
            left_axis=ft.ChartAxis(
                labels_size=40,
                labels_interval=y_interval,
            ),
            bottom_axis=ft.ChartAxis(
                labels_interval=x_interval,
                labels_size=40,
            ),
        )

        # 定义折线图的数据系列
        line_chart = ft.LineChartData(
            color=self.color,
            stroke_width=2,
            curved=True,  # 使线条平滑
            stroke_cap_round=True,
            # 设置线条下方的渐变效果
            below_line_gradient=ft.LinearGradient(
                begin=ft.alignment.top_center,
                end=ft.alignment.bottom_center,
                colors=[
                    ft.colors.with_opacity(0.25, self.color),
                    "transparent",
                ],
            ),
        )

        # 为折线图数据系列添加数据点
        line_chart.data_points = [self.create_data_point(x, y) for x, y in self.data]
        chart.data_series = [line_chart]
        return chart

    # 创建数据点的方法
    def create_data_point(self, x, y):
        return ft.LineChartDataPoint(
            x,
            y,
            selected_below_line=ft.ChartPointLine(
                width=0.5, color="white54", dash_pattern=[2, 4]
            ),
            selected_point=ft.ChartCirclePoint(stroke_width=1),
        )

    # 构建UI组件的方法
    def build(self):
        self.content = ft.Column(
            horizontal_alignment="center",
            controls=[
                ft.Text(
                    f"Yearly Historical Prices for {self.label}",
                    size=16,
                    weight="bold",
                ),
                self.chart,
            ],
        )
