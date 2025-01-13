import base64
import os
import cv2
from flet import *
import flet as ft
from tqdm import tqdm
from ultralytics import YOLO
import shutil
import copy
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import onnxruntime
from assets.funticon.onnxDetect import *
from assets.funticon.plate_ocr import *

# 车牌字符集  
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZOI了子桂日曰险品"  

# 替换规则  
replace_dict = {  
    "了": "辽",  
    "子": "辽",  
    "O": "0",  
    "I": "1",  
    "桂": "挂",  
    "日": "B",  
    "曰": "B" ,
    "皖": "辽B" ,
    "辽8": "辽B" ,
    "晋": "辽B"  
} 
device = 0  # ''cpu' or 0


# 初始化 ONNX 模型
def initialize_onnx_models(detect_model_path, rec_model_path):
    providers = ["CPUExecutionProvider"]
    session_detect = onnxruntime.InferenceSession(
        detect_model_path, providers=providers
    )
    session_rec = onnxruntime.InferenceSession(rec_model_path, providers=providers)
    return session_detect, session_rec


def allFilePath(rootPath, allFIleList):  # 遍历文件
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


def find_image_files(directory):
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
    )
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(root, file)
                absolute_path = os.path.abspath(file_path)
                image_files.append(absolute_path)
    return image_files


def find_onnx_files(directory):
    onnx_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".onnx"):
                onnx_files.append(os.path.join(root, file))
    return onnx_files


def create_shadow():
    return [
        ft.BoxShadow(
            offset=ft.Offset(19, 19),
            blur_radius=55,
            color="#ccccce",
            blur_style=ft.ShadowBlurStyle.NORMAL,
        ),
        ft.BoxShadow(
            offset=ft.Offset(-19, -19),
            blur_radius=55,
            color="#ffffff",
            blur_style=ft.ShadowBlurStyle.NORMAL,
        ),
    ]


def create_container(content, bgcolor, margin, padding, shadow_blur=5):
    return ft.Container(
        content=content,
        border_radius=15,
        expand=True,
        bgcolor=bgcolor,
        margin=margin,
        padding=padding,
        shadow=[
            ft.BoxShadow(
                blur_radius=shadow_blur,
                color="#ccccce",
                offset=ft.Offset(5, 5),
                blur_style=ft.ShadowBlurStyle.NORMAL,
            ),
            ft.BoxShadow(
                blur_radius=shadow_blur,
                color="#ffffff",
                offset=ft.Offset(-5, -5),
                blur_style=ft.ShadowBlurStyle.NORMAL,
            ),
        ],
    )


def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


class MainView(ft.Container):
    def __init__(self, page: ft.Page, col):
        super().__init__()
        self.col = col
        self.shadow = create_shadow()
        self.image_viewer = ft.GridView(
            expand=1,
            runs_count=150,
            max_extent=350,
            child_aspect_ratio=1.0,
            spacing=5,
            run_spacing=5,
            auto_scroll=True,
            reverse=True,
        )
        self.video_player = ft.Column()
        self.tabs = ft.Tabs(
            animation_duration=0,
            tab_alignment=ft.TabAlignment.START,
            scrollable=True,
            unselected_label_color="grey",
            expand=True,
            divider_height=0,
            on_change=self.on_change,
            tabs=[
                ft.Tab(
                    tab_content=ft.Text("结果展示", size=12, weight="bold"),
                    content=ft.Container(
                        expand=True,
                        bgcolor="#e1e2e8",
                        margin=ft.margin.only(bottom=10),
                        border_radius=10,
                        content=self.video_player,
                    ),
                ),
                ft.Tab(
                    tab_content=ft.Text("多图片展示", size=12, weight="bold"),
                    content=ft.Container(
                        expand=True,
                        bgcolor="#e1e2e8",
                        margin=ft.margin.only(bottom=10),
                        border_radius=10,
                        content=ft.Column([self.image_viewer]),
                        on_hover=self.on_hover_image_viewer_stop,
                    ),
                ),
            ],
        )

        self.content = ft.Column(
            controls=[
                create_container(
                    content=self.tabs,
                    bgcolor="#f3f3f3",
                    margin=ft.margin.all(10),
                    padding=ft.padding.symmetric(horizontal=15, vertical=5),
                )
            ],
            spacing=5,
        )

    def did_mount(self):
        self.page.session.set("video_player", self.video_player.uid)
        self.page.session.set("image_viewer", self.image_viewer.uid)
        self.page.session.set("main_tabs", self.tabs.uid)

    def on_change(self, e):
        self.image_viewer.auto_scroll = True
        self.image_viewer.update()

    def on_hover_image_viewer_stop(self, e):
        if self.tabs.selected_index == 1:
            self.image_viewer.auto_scroll = not self.image_viewer.auto_scroll
        self.image_viewer.update()


class RightView(ft.Container):
    def __init__(self, page: ft.Page, col):
        super().__init__()
        self.col = col
        self.shadow = create_shadow()

        def pick_files_model(e: ft.FilePickerResultEvent):
            file = ", ".join(map(lambda f: f.name, e.files)) if e.files else None
            if not file:
                return
            self.drop_menu.value = file
            self.drop_menu.options.append(
                dropdown.Option(
                    file,
                    text_style=TextStyle(size=14, weight="bold", font_family="Arian"),
                )
            )
            self.drop_menu.value = file
            self.drop_menu.update()

        def pick_files_path(self, e: ft.FilePickerResultEvent):
            if not e.files:  # print(i)
                return
            current_file = e.files[0].path
            page.session.set("current_file", current_file)
            self.video_control.controls.clear()
            video_files = []
            for i in [current_file]:
                video_files.append(VideoMedia(i))
            self.video_control.controls.append(
                ft.Video(
                    playlist=video_files,
                    expand=True,
                    playlist_mode=ft.PlaylistMode.NONE,
                    fill_color=ft.Colors.TRANSPARENT,
                    fit=ImageFit.FILL,
                    aspect_ratio=16 / 9,
                    volume=100,
                    autoplay=False,
                    filter_quality=ft.FilterQuality.HIGH,
                    muted=False,
                )
            )
            self.video_control.update()

            self.main_tabs.selected_index = 0
            self.main_tabs.update()

        pick_model = ft.FilePicker(on_result=pick_files_model)
        pick_detect = ft.FilePicker(on_result=lambda e: pick_files_path(self, e))
        self.drop_menu = ft.Dropdown(
            value="best.pt",
            dense=True,
            expand=True,
            filled=True,
            options=[
                dropdown.Option(
                    "best.pt",
                    text_style=TextStyle(size=14, weight="bold", font_family="Arian"),
                )
            ],
        )
        self.img_maxnum = Column([], expand=True)
        self.img_num = Column([], expand=True)
        page.overlay.append(pick_model)
        page.overlay.append(pick_detect)
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tab_alignment=ft.TabAlignment.CENTER,
            scrollable=True,
            expand=True,
            unselected_label_color="grey",
            on_change=self.on_change,
            tabs=[
                ft.Tab(
                    tab_content=ft.Text("默认配置", size=12, weight="bold"),
                    content=ft.Container(
                        expand=True,
                        content=ft.Column(
                            [
                                ft.Row(
                                    [
                                        self.drop_menu,
                                        ft.IconButton(
                                            icon=ft.Icons.FOLDER_OPEN,
                                            hover_color=Colors.TRANSPARENT,
                                            highlight_color=Colors.TRANSPARENT,
                                            icon_size=35,
                                            on_click=lambda _: pick_model.pick_files(
                                                allow_multiple=True
                                            ),
                                        ),
                                    ],
                                    spacing=0,
                                ),
                                imgs := ft.TextField(
                                    value="./images",
                                    label="图片文件夹路径",
                                    border=ft.InputBorder.NONE,
                                    filled=True,
                                ),
                                Row(
                                    [
                                        ft.FilledButton(
                                            "上传文件",
                                            style=ButtonStyle(
                                                shape=RoundedRectangleBorder(5)
                                            ),
                                            expand=True,
                                            on_click=lambda _: pick_detect.pick_files(
                                                allow_multiple=False,
                                            ),
                                        )
                                    ]
                                ),
                                ft.Text("置信度:"),
                                conf := ft.Slider(
                                    value=25,
                                    label="{value}%",
                                    max=100.0,
                                    min=0.0,
                                    divisions=20,on_change=self.slider_changed,
                                ),
                                ft.Text("IOU:"),
                                iou := ft.Slider(
                                    value=75,
                                    label="{value}%",
                                    max=100.0,
                                    min=0.0,
                                    divisions=20,on_change=self.slider_changed,
                                ),
                                Row(
                                    [
                                        ft.FilledButton(
                                            "推理",
                                            height=50,
                                            style=ButtonStyle(
                                                shape=RoundedRectangleBorder(5)
                                            ),
                                            expand=True,
                                            on_click=lambda e: self.on_detect(e),
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ),
                ),
                ft.Tab(
                    tab_content=ft.Text("多图片推理", size=12, weight="bold"),
                    content=ft.Column(
                        [
                            Text("放大号:", size=12, weight="bold"),
                            Row([self.img_maxnum], height=90, expand=False),
                            Text("车牌:", size=12, weight="bold"),
                            Row([self.img_num], height=90, expand=False),
                            Container(
                                output := ListView(
                                    spacing=5, auto_scroll=True, divider_thickness=1
                                ),
                                expand=True,
                                border=border.all(2, color=Colors.GREY_200),
                                border_radius=10,
                                on_hover=self.on_hover_ListView_stop,
                            ),
                            Column(
                                [
                                    Row(
                                        [
                                            ft.FilledButton(
                                                "多图片播放",
                                                height=50,
                                                style=ButtonStyle(
                                                    shape=RoundedRectangleBorder(5)
                                                ),
                                                expand=True,
                                                on_click=lambda e: self.Model_Detect(e),
                                            ),
                                        ]
                                    ),
                                    Row(
                                        [
                                            ft.FilledButton(
                                                "推理",
                                                height=50,
                                                style=ButtonStyle(
                                                    shape=RoundedRectangleBorder(5)
                                                ),
                                                expand=True,
                                                on_click=lambda e: self.Model_Detect(e),
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ),
            ],
        )





        current_directory = os.getcwd()
        onnx_folder = os.path.join(current_directory, "onnx")
        onnx_files = find_onnx_files(onnx_folder)

        for file in onnx_files:
            self.directory, filename = os.path.split(file)
            self.drop_menu.options.append(
                dropdown.Option(
                    filename,
                    text_style=TextStyle(size=14, weight="bold", font_family="Arian"),
                )
            )

        self.imgs = imgs
        self.iou = iou
        self.conf = conf
        self.output = output
        # 初始化 ONNX 模型
        detect_model_path = "onnx/plate_detect.onnx"
        rec_model_path = "onnx/plate_rec_color.onnx"
        self.session_rec = initialize_onnx_models(
            detect_model_path, rec_model_path
        )  # 初始化 session_rec
        content = ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(
                            "模型设置",
                            size=16,
                            weight="bold",
                            font_family="muli",
                        )
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                self.tabs,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            expand=True,
        )

        self.content = ft.Column(
            controls=[
                create_container(
                    content=content,
                    bgcolor="#f3f3f3",
                    margin=ft.margin.all(10),
                    padding=ft.padding.symmetric(horizontal=15, vertical=10),
                )
            ],
            spacing=5,
        )
    def slider_changed(self,e):
        slider_value = e.control.value
        result = int(slider_value)
        e.control.label = f"{result}%"
        e.control.update()
    def did_mount(self):
        self.video_control = self.page.get_control(
            self.page.session.get("video_player")
        )
        self.image_viewer = self.page.get_control(self.page.session.get("image_viewer"))
        self.main_tabs = self.page.get_control(self.page.session.get("main_tabs"))

    def on_hover_ListView_stop(self, e):
        if self.tabs.selected_index == 1:
            self.output.auto_scroll = not self.output.auto_scroll
        self.output.update()

    def on_change(self, e):
        self.output.auto_scroll = True
        self.output.update()

    def click_image_view(self, e):
        self.main_tabs.selected_index = 0
        self.main_tabs.update()
        self.video_control.controls.clear()
        src = e.control.content.controls[0].src
        video_files = []
        for i in [src]:
            video_files.append(VideoMedia(i))
        self.video_control.controls.append(
            ft.Video(
                expand=True,
                playlist=video_files,
                playlist_mode=ft.PlaylistMode.NONE,
                fill_color=ft.Colors.TRANSPARENT,
                fit=ImageFit.FILL,
                aspect_ratio=16 / 9,
                volume=100,
                autoplay=False,
                filter_quality=ft.FilterQuality.HIGH,
                muted=False,
            )
        )
        self.video_control.update()

    def Model_Detect(self, e):

        # 指定你要搜索的目录
        target_directory = "./result"
        if e.control.text == "推理":
            e.control.disabled = True
            e.control.text = "Loading"
            e.control.style = ButtonStyle(
                bgcolor=Colors.GREY_500, shape=RoundedRectangleBorder(5)
            )
            e.control.update()
            self.main_tabs.selected_index = 1
            self.main_tabs.update()
            model_file = "./onnx/" + self.drop_menu.value
            input_imgs_folder = self.imgs.value
            conf_thresh = float(self.conf.value / 100)
            iou_thresh = float(self.iou.value / 100)

            image_files = find_image_files(target_directory)

            if len(self.image_viewer.controls) > 0:
                self.image_viewer.controls.clear()

            self.detect(
                model_file,
                input_imgs_folder,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
            )

            e.control.disabled = False
            e.control.text = "推理"
            e.control.style = ButtonStyle(shape=RoundedRectangleBorder(5))
            e.control.update()
        elif e.control.text == "多图片播放":
            self.output.value = None
            self.output.update()
            image_files = find_image_files(target_directory)
            self.video_control.controls.clear()
            self.main_tabs.selected_index = 0
            self.main_tabs.update()
            video_files = []
            for i in image_files:

                video_files.append(VideoMedia(i))

            self.video_control.controls.append(
                ft.Video(
                    expand=True,
                    playlist=video_files,
                    playlist_mode=ft.PlaylistMode.NONE,
                    fill_color=ft.Colors.TRANSPARENT,
                    fit=ImageFit.FILL,
                    aspect_ratio=16 / 9,
                    volume=100,
                    autoplay=False,
                    filter_quality=ft.FilterQuality.HIGH,
                    muted=False,
                )
            )
            self.video_control.update()

    def on_detect(self, e):

        current_file = self.page.session.get("current_file")
        conf_thresh = float(self.conf.value / 100)
        iou_thresh = float(self.iou.value / 100)

        def is_video_file(file_path):
            video_extensions = [
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".wmv",
                ".flv",
                ".mpeg",
                ".mpg",
            ]
            # 提取文件扩展名
            ext = file_path.split(".")[-1].lower()
            # 判断扩展名是否在视频扩展名列表中
            return f".{ext}" in video_extensions

        if is_video_file(current_file):
            e.control.disabled = True
            e.control.text = "Loading"
            e.control.style = ButtonStyle(
                bgcolor=Colors.GREY_500, shape=RoundedRectangleBorder(5)
            )
            e.control.update()
            self.Video_out(conf_thresh, iou_thresh)
            self.video_control.controls.clear()
            save_img_path = "assets/result/output.mp4"

            self.main_tabs.selected_index = 0
            self.main_tabs.update()
            video_files = []
            for i in [save_img_path]:
                video_files.append(VideoMedia(i))

            self.video_control.controls.append(
                ft.Video(
                    expand=True,
                    playlist=video_files,
                    playlist_mode=ft.PlaylistMode.NONE,
                    fill_color=ft.Colors.TRANSPARENT,
                    fit=ImageFit.FILL,
                    aspect_ratio=16 / 9,
                    volume=100,
                    autoplay=False,
                    filter_quality=ft.FilterQuality.HIGH,
                    muted=False,
                )
            )
            self.video_control.update()
            e.control.disabled = False
            e.control.text = "推理"
            e.control.style = ButtonStyle(shape=RoundedRectangleBorder(5))
            e.control.update()
        else:

            model_file = "./onnx/" + self.drop_menu.value
            # input_imgs_folder = self.imgs.value

            output_dir = "assets/result"
            file_list = [current_file]
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            save_path = output_dir
            clear_directory(save_path)
            count = 0
            model = YOLO(model_file)

            for pic_ in tqdm(file_list, desc="处理图片:"):
                img_name = os.path.basename(pic_)
                save_img_path = os.path.join(
                    save_path, f"{conf_thresh}_{iou_thresh}" + img_name
                )
                count += 1
                img = cv2.imread(pic_)
                img0 = copy.deepcopy(img)
                results = model.predict(
                    img,
                    save=False,
                    imgsz=640,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    device=device,
                    verbose=False,
                )

                cls_name = results[0].names
                boxes = results[0].boxes.data.tolist()
                height_area = 80
                for result in results:
                    img = result.plot(font_size=10)
                    result.save(save_img_path)
                for box in boxes:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    conf = box[4]
                    cls = cls_name[int(box[5])]
                    if cls == "number":
                        cropped_img = img0[y1 - 10 : y2 + 10, x1 - 10 : x2 + 10]
                        # ocr_result = ocr.ocr(cropped_img, inv=True, cls=True)
                        try:
                            if ocr_result[0]:
                                ocr_result = ocr_result[0]
                                txts = [line[1][0] for line in ocr_result]
                                scores = [line[1][1] for line in ocr_result]
                                if len("".join(txts)) > 3:
                                    img = draw_box_text(
                                        img,
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        txts,
                                        height_area,
                                        (0, 255, 0),
                                    )

                        except Exception as e:
                            print(f"Error occurred: {e}")
                    cv2.imwrite(save_img_path, img)
                self.video_control.controls.clear()

                self.main_tabs.selected_index = 0
                self.main_tabs.update()
                video_files = []
                for i in [save_img_path]:
                    video_files.append(VideoMedia(i))

                self.video_control.controls.append(
                    ft.Video(
                        expand=True,
                        playlist=video_files,
                        playlist_mode=ft.PlaylistMode.NONE,
                        fill_color=ft.Colors.TRANSPARENT,
                        fit=ImageFit.FILL,
                        aspect_ratio=16 / 9,
                        volume=100,
                        autoplay=False,
                        filter_quality=ft.FilterQuality.HIGH,
                        muted=False,
                    )
                )
                self.video_control.update()

    def detect(
        self,
        model_file,
        input_imgs_folder,
        output_dir="result",
        conf_thresh=0.25,
        iou_thresh=0.75,
    ):

        file_list = []
        allFilePath(input_imgs_folder, file_list)
        providers = ["CUDAExecutionProvider"]
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        img_size = (640, 640)
        session_detect = onnxruntime.InferenceSession(
            "onnx/plate_detect.onnx", providers=providers
        )
        session_rec = onnxruntime.InferenceSession(
            "onnx/plate_rec_color.onnx", providers=providers
        )
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        save_path = output_dir
        clear_directory(save_path)
        count = 0
        model = YOLO(model_file)

        for pic_ in tqdm(file_list, desc="处理图片:"):
            with open(pic_, "rb") as f:
                img_bytes = f.read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img0 = copy.deepcopy(img)
            img, r, left, top = detect_pre_precessing(img, img_size)  # 检测前处理
            y_onnx = session_detect.run(
                [session_detect.get_outputs()[0].name],
                {session_detect.get_inputs()[0].name: img},
            )[0]

            img_name = os.path.basename(pic_)
            count += 1
            img = cv2.imread(pic_)
            img0 = copy.deepcopy(img)
            results = model.predict(
                img,
                save=False,
                imgsz=640,
                conf=conf_thresh,
                iou=iou_thresh,
                device=device,
                verbose=False,
            )
            # 处理预测结果
            img_name = os.path.basename(pic_)
            save_img_path = os.path.join(
                save_path, f"{conf_thresh}_{iou_thresh}" + img_name
            )
            cls_name = results[0].names
            boxes = results[0].boxes.data.tolist()
            height_area = 80
            plate_no_txt, Number_txt = "车牌号：\n", "放大号：\n"
            self.img_num.controls.clear()
            self.img_maxnum.controls.clear()
            for result in results:
                img = result.plot(font_size=6, labels=False)
                result.save(save_img_path)
            for box in boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                conf = box[4]
                cls = cls_name[int(box[5])]

                if cls == "MPnumber":
                    # 检查裁剪区域是否在图像范围内
                    if x1 - 15 < 0 or y1 - 17 < 0 or x2 + 15 > img0.shape[1] or y2 + 15 > img0.shape[0]:
                        print(f"裁剪区域超出图像范围: {pic_}, box: {box}")
                        continue

                    cropped_img_number = img0[y1 - 17: y2 + 15, x1 - 15: x2 + 15]

                    # 检查裁剪后的图像是否为空
                    if cropped_img_number.size == 0:
                        print(f"裁剪后的图像为空: {pic_}, box: {box}")
                        continue

                    # 将图片编码为PNG格式
                    _, buffer = cv2.imencode(".png", cropped_img_number)
                    # 将图片编码为Base64字符串
                    base64_str = base64.b64encode(buffer).decode("utf-8")

                    try:
                        model_path = './onnx/model.onnx'
                        vocab_path = './assets/vocab.txt'
                        recognized_text = recognize_image(base64_str, model_path, vocab_path)[0].replace(" ", "")
                        recognized_text = ''.join(
                            [replace_dict.get(char, char) for char in recognized_text if char in plateName]).replace(" ",
                                                                                                                    "")
                        # print("识别出的字符拼接结果：", recognized_text)
                    except Exception as e:
                        print("OCR识别失败:", str(e))
                    if len(recognized_text) > 3:
                        draw_color = (255, 0, 255)
                        img = draw_box_text(
                            img,
                            x1,
                            y1,
                            x2,
                            y2,
                            recognized_text,
                            height_area,
                            draw_color,
                        )
                        Number_txt = f'放大号：{"".join(recognized_text)}\n'
                    self.img_maxnum.controls.append(
                        ft.Image(src_base64=base64_str, fit=ImageFit.FILL, expand=True)
                    )

                if cls == "double_plate":
                    # 检查裁剪区域是否在图像范围内
                    if x1 - 0 < 0 or y1 - 0 < 0 or x2 + 8 > img0.shape[1] or y2 + 8 > img0.shape[0]:
                        print(f"裁剪区域超出图像范围: {pic_}, box: {box}")
                        continue

                    cropped_img_plate = img0[y1 - 0: y2 + 8, x1 - 0: x2 + 8]
                    _, buffer = cv2.imencode(".png", cropped_img_plate)
                    # 将图片编码为Base64字符串
                    base64_str = base64.b64encode(buffer).decode("utf-8")

                    outputs = post_precessing(y_onnx, r, left, top)  # 检测后处理
                    result_list = rec_plate(outputs, img0, session_rec)
                    result_str, ori_img = draw_result(cropped_img_plate, result_list)
                    if len("".join(result_str)) > 3:
                        draw_color = (0, 255, 255)
                        img = draw_box_text(
                            img, x1, y1, x2, y2, result_str, height_area, draw_color
                        )
                        plate_no_txt = f'车牌号：{"".join(result_str)}\n'
                    self.img_num.controls.append(
                        ft.Image(src_base64=base64_str, fit=ImageFit.FILL, expand=True)
                    )
                cv2.imwrite(save_img_path, img)

            self.output.controls.append(
                ListTile(
                    leading=Text(
                        f"no.{count}", size=14, weight="bold", font_family="Arian"
                    ),
                    title=Column(
                        [
                            Text(f"{pic_}", size=9, font_family="Arian"),
                            Text(f"{Number_txt}{plate_no_txt}", size=13, weight="bold"),
                        ],
                        spacing=0,
                    ),
                )
            )

            self.image_viewer.controls.append(
                ft.Container(
                    Column(
                        [
                            ft.Image(
                                src=save_img_path,
                                repeat=ft.ImageRepeat.NO_REPEAT,
                                fit=ImageFit.FILL,
                                border_radius=ft.border_radius.all(10),
                                expand=True,
                            ),
                            Text(count, font_family="Arian", size=12),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                    ),
                    margin=5,
                    on_click=lambda e: self.click_image_view(e),
                )
            )
            self.output.update()
            self.img_num.update()
            self.img_maxnum.update()
            self.image_viewer.update()

    def Video_out(self, conf_thresh=0.25, iou_thresh=0.75):
        current_file = self.page.session.get("current_file")
        model_file = "./onnx/" + self.drop_menu.value
        output_dir = "assets/result"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        clear_directory(output_dir)
        model = YOLO(model_file)
        cap = cv2.VideoCapture(current_file)
        if not cap.isOpened():
            raise IOError("Cannot open video")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join(output_dir, "output.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        for _ in tqdm(range(total_frames), desc="Processing", unit="frames"):
            success, frame = cap.read()
            if not success:
                break

            results = model.predict(
                frame,
                save=False,
                imgsz=640,
                conf=conf_thresh,
                iou=iou_thresh,
                device=device,
                verbose=False,
            )
            cls_name = results[0].names
            boxes = results[0].boxes.data.tolist()
            height_area = 80
            for box in boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cls = cls_name[int(box[5])]
                if cls == "number":
                    cropped_img = frame[y1 - 10 : y2 + 10, x1 - 10 : x2 + 10]
                    # ocr_result = ocr.ocr(cropped_img, inv=True, cls=True)
                    try:
                        if ocr_result[0]:
                            ocr_result = ocr_result[0]
                            txts = [line[1][0] for line in ocr_result]
                            # scores = [line[1][1] for line in ocr_result]
                            if len("".join(txts)) > 3:
                                draw_box_text(
                                    frame,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    txts,
                                    height_area,
                                    (0, 255, 0),
                                )
                    except Exception as e:
                        print(f"Error occurred: {e}")
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()


def draw_box_text(img, x1, y1, x2, y2, txts, height_area, draw_color):
    # 将图像转换为PIL图像以便绘制文字
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    fontText = ImageFont.truetype(
        "assets/fonts/platech.ttf",
        height_area,
        encoding="utf-8",
    )

    draw.text(
        (x1, y1 + 150),
        "".join(txts),
        draw_color,
        font=fontText,
    )
    img_pil = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_pil


def main(page: ft.Page):
    try:
        page.window.always_on_top = True
        page.window.center()
        page.bgcolor = "#F0F0F3"
        page.window.width = 1200
        page.window.height = 800
        page.fonts = {"muli": "/fonts/Muli-Regular.ttf"}
        page.title = "Yolo检测"

        page.theme = ft.Theme(
            scrollbar_theme=ft.ScrollbarTheme(
                thumb_color=ft.Colors.TRANSPARENT,
                track_color=ft.Colors.TRANSPARENT,
                track_border_color=ft.Colors.TRANSPARENT
            ),font_family="微软雅黑"
        ) # 隐藏滚动条
        page.add(
            ft.ResponsiveRow(
                [
                    MainView(page, 9),
                    RightView(page, 3),
                ],
                expand=True,
            )
        )
    except Exception as e:
        print(e)


ft.app(target=main, assets_dir="assets;onnx;result;images")
