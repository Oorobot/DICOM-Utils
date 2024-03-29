import argparse
import json
import os

import matplotlib
import numpy as np
import SimpleITK as sitk
import vtk

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPiecewiseFunction,
    vtkPolyData,
    vtkPolygon,
)

# noinspection PyUnresolvedReferences
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersSources import vtkPlaneSource
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkFollower,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingFreeType import vtkVectorText
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper

from utils.dicom import suvbw2image
from utils.utils import LABELS


def flipXY(array):
    array = np.flip(array, axis=1)
    array = np.flip(array, axis=2)
    return array


def flip_boxXY(box, size):
    for b in box:
        b[0], b[3] = size[0] - b[3], size[0] - b[0]  # X
        b[1], b[4] = size[1] - b[4], size[1] - b[1]  # Y
    return box


# -----------------------------------------------------------#
#                       自定义键盘交互
# -----------------------------------------------------------#
class KeyPressStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        super().__init__()
        self.AddObserver("KeyPressEvent", self.OnKeyPress)

    def OnKeyPress(self, obj, event):
        print(f"Key {obj.GetInteractor().GetKeySym()} pressed")
        camera = obj.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera()
        if obj.GetInteractor().GetKeySym() == "w":
            direction_of_projection = camera.GetDirectionOfProjection()
            position = camera.GetPosition()
            new_position = np.array(position) + 100 * np.array(direction_of_projection)
            new_focal_point = new_position + 400 * np.array(direction_of_projection)
            camera.SetPosition(*new_position.tolist())
            camera.SetFocalPoint(*new_focal_point.tolist())
        if obj.GetInteractor().GetKeySym() == "s":
            direction_of_projection = camera.GetDirectionOfProjection()
            position = camera.GetPosition()
            new_position = np.array(position) - 100 * np.array(direction_of_projection)
            new_focal_point = new_position + 400 * np.array(direction_of_projection)
            camera.SetPosition(*new_position.tolist())
            camera.SetFocalPoint(*new_focal_point.tolist())


# -----------------------------------------------------------#
#                           平面
# -----------------------------------------------------------#
# def vtk_plane():
#     # Setup four points
#     points = vtkPoints()
#     point_x1 = origin[0] - spacing[0] * size[0] * 0.05
#     point_x2 = origin[0] + spacing[0] * size[0] * 1.05
#     point_y1 = origin[1] - spacing[1] * size[1] * 0.05
#     point_y2 = origin[1] + spacing[1] * size[1] * 1.05
#     point_z = origin[2] + spacing[2] * 194
#     points.InsertNextPoint(point_x1, point_y1, point_z)
#     points.InsertNextPoint(point_x1, point_y2, point_z)
#     points.InsertNextPoint(point_x2, point_y2, point_z)
#     points.InsertNextPoint(point_x2, point_y1, point_z)

#     # point_x = origin[0] + spacing[0] * 83
#     # point_y1 = origin[1] + spacing[1] * size[1] * 1.05
#     # point_y2 = origin[1] - spacing[1] * size[1] * 0.05
#     # point_z1 = origin[2] + spacing[2] * size[2] * 1.05
#     # point_z2 = origin[2] - spacing[2] * size[2] * 0.05
#     # points.InsertNextPoint(point_x, point_y1, point_z1)
#     # points.InsertNextPoint(point_x, point_y2, point_z1)
#     # points.InsertNextPoint(point_x, point_y2, point_z2)
#     # points.InsertNextPoint(point_x, point_y1, point_z2)

#     # Create the polygon
#     polygon = vtkPolygon()
#     polygon.GetPointIds().SetNumberOfIds(4)  # make a quad
#     polygon.GetPointIds().SetId(0, 0)
#     polygon.GetPointIds().SetId(1, 1)
#     polygon.GetPointIds().SetId(2, 2)
#     polygon.GetPointIds().SetId(3, 3)

#     # Add the polygon to a list of polygons
#     polygons = vtkCellArray()
#     polygons.InsertNextCell(polygon)

#     # Create a PolyData
#     polygonPolyData = vtkPolyData()
#     polygonPolyData.SetPoints(points)
#     polygonPolyData.SetPolys(polygons)

#     # Create a mapper and actor
#     mapper = vtkPolyDataMapper()
#     mapper.SetInputData(polygonPolyData)

#     actor = vtkActor()
#     actor.SetMapper(mapper)
#     color = (0, 0, 0) if "pet" in args.vis else (0, 77, 153)
#     actor.GetProperty().SetColor(color)
#     actor.GetProperty().SetOpacity(0.25)
#     renderer.AddActor(actor)


# -----------------------------------------------------------#
#                     标注框以及文字
# -----------------------------------------------------------#
def vtk_bounding_boxes(boxes, origin, spacing, texts, colors, size, up=True):
    origin = np.array(origin)
    spacing = np.array(spacing)
    actors = []
    for box, text, color in zip(boxes, texts, colors):
        # 将索引号转换为坐标位置
        point1, point2 = np.array(box[0:3]), np.array(box[3:6])
        x1, y1, z1 = origin + point1 * spacing
        x2, y2, z2 = origin + point2 * spacing
        position = [
            [x1, y1, z1],
            [x1, y2, z1],
            [x2, y2, z1],
            [x2, y1, z1],
            [x1, y1, z2],
            [x1, y2, z2],
            [x2, y2, z2],
            [x2, y1, z2],
        ]

        # 边界框
        points = vtkPoints()  # 点
        points.SetNumberOfPoints(8)
        for i in range(8):
            points.SetPoint(i, *position[i])
        lines = vtkCellArray()  # 线
        lines.InsertNextCell(5)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(3)
        lines.InsertCellPoint(0)
        lines.InsertNextCell(5)
        lines.InsertCellPoint(4)
        lines.InsertCellPoint(5)
        lines.InsertCellPoint(6)
        lines.InsertCellPoint(7)
        lines.InsertCellPoint(4)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(4)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(1)
        lines.InsertCellPoint(5)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(2)
        lines.InsertCellPoint(6)
        lines.InsertNextCell(2)
        lines.InsertCellPoint(3)
        lines.InsertCellPoint(7)

        cube = vtkPolyData()  # 多边形
        cube.SetPoints(points)
        cube.SetLines(lines)

        cubeMapper = vtkPolyDataMapper()
        cubeMapper.SetInputData(cube)
        cubeMapper.Update()

        bbActor = vtkActor()
        bbActor.SetMapper(cubeMapper)
        bbActor.GetProperty().SetColor(*color)
        actors.append(bbActor)

        # 文本框
        vtk_text = vtkVectorText()
        vtk_text.SetText(text)
        textMapper = vtkPolyDataMapper()
        textMapper.SetInputConnection(vtk_text.GetOutputPort())
        textActor = vtkFollower()
        textActor.SetMapper(textMapper)
        textActor.SetScale(size / 30, size / 30, size / 30)
        textActor.AddPosition(x2, y2, z2 if up else z1)
        textActor.GetProperty().SetColor(*color)
        actors.append(textActor)
    return actors


# -----------------------------------------------------------#
#                            体素
# -----------------------------------------------------------#
def ct_vtk_volume(ct_array: np.ndarray, origin: list, spacing: list, body_array=None):
    # ct_array = np.clip(ct_array, -450, 1050)
    if body_array is not None:
        ct_array = ct_array * body_array + (1 - body_array) * -1000

    ct_array = sitk.GetArrayFromImage(sitk.GetImageFromArray(ct_array.astype(np.float32)))

    # 创建 vtk 图像
    vtk_image = vtkImageImportFromArray()
    vtk_image.SetArray(ct_array)
    vtk_image.SetDataSpacing(spacing)
    vtk_image.SetDataOrigin(origin)
    vtk_image.Update()

    # 体积将通过光线投射alpha合成显示。
    # 需要光线投射映射器来进行光线投射。
    volume_mapper = vtkFixedPointVolumeRayCastMapper()
    volume_mapper.SetInputConnection(vtk_image.GetOutputPort())

    # 使用颜色转换函数，对不同的值设置不同的函数
    volume_color = vtkColorTransferFunction()
    cmap = matplotlib.colormaps["gray"]
    colors = cmap(np.linspace(0, 1, cmap.N))
    rgb_point = np.linspace(-450, 1050, 256)
    for i in range(0, 256):
        volume_color.AddRGBPoint(rgb_point[i], *colors[i, :3])

    # 使用透明度转换函数，用于控制不同组织之间的透明度
    volume_scalar_opacity = vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.00)
    volume_scalar_opacity.AddPoint(200, 0.25)
    volume_scalar_opacity.AddPoint(500, 0.45)
    volume_scalar_opacity.AddPoint(1000, 0.65)
    volume_scalar_opacity.AddPoint(1150, 0.90)
    # volume_scalar_opacity.AddPoint(100, 0.15)
    # volume_scalar_opacity.AddPoint(500, 0.40)
    # volume_scalar_opacity.AddPoint(1000, 0.65)
    # volume_scalar_opacity.AddPoint(1150, 0.90)

    # 梯度不透明度函数用于降低体积“平坦”区域的不透明度，同时保持组织类型之间边界的不透明度。梯度是以强度在单位距离上的变化量来测量的
    volume_gradient_opacity = vtkPiecewiseFunction()
    volume_gradient_opacity.AddPoint(0, 0.0)
    volume_gradient_opacity.AddPoint(90, 0.5)
    volume_gradient_opacity.AddPoint(100, 1.0)

    volume_property = vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.SetGradientOpacity(volume_gradient_opacity)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()
    volume_property.SetAmbient(0.4)
    volume_property.SetDiffuse(0.6)
    volume_property.SetSpecular(0.2)

    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume


# -----------------------------------------------------------#
#                           体素
# -----------------------------------------------------------#
def suv_vtk_volume(suv_array: np.ndarray, origin: list, spacing: list):
    array = suvbw2image(suv_array, 2.5)
    suv = sitk.GetArrayFromImage(sitk.GetImageFromArray(array))

    # 创建 vtk 图像
    vtk_image = vtkImageImportFromArray()
    vtk_image.SetArray(suv)
    vtk_image.SetDataSpacing(spacing)
    vtk_image.SetDataOrigin(origin)
    vtk_image.Update()

    # 体积将通过光线投射alpha合成显示。
    # 需要光线投射映射器来进行光线投射。
    volume_mapper = vtkFixedPointVolumeRayCastMapper()
    volume_mapper.SetInputConnection(vtk_image.GetOutputPort())

    # 使用颜色转换函数，对不同的值设置不同的函数
    volume_color = vtkColorTransferFunction()
    linspace = np.linspace(0, 1, 256)
    hot_cmap = matplotlib.colormaps["hot"]
    hot_colors = hot_cmap(linspace)
    for i in range(0, 256):
        volume_color.AddRGBPoint(linspace[i], *hot_colors[i, 0:3])

    # 使用透明度转换函数，用于控制不同组织之间的透明度
    volume_scalar_opacity = vtkPiecewiseFunction()
    for i in range(10):
        volume_scalar_opacity.AddPoint(i / 10, i / 10)

    # 梯度不透明度函数用于降低体积“平坦”区域的不透明度，同时保持组织类型之间边界的不透明度。梯度是以强度在单位距离上的变化量来测量的
    volume_gradient_opacity = vtkPiecewiseFunction()
    volume_gradient_opacity.AddPoint(0, 0)
    volume_gradient_opacity.AddPoint(0.02, 0.5)
    volume_gradient_opacity.AddPoint(0.03, 1.0)

    volume_property = vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.SetGradientOpacity(volume_gradient_opacity)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()
    volume_property.SetAmbient(0.4)
    volume_property.SetDiffuse(0.6)
    volume_property.SetSpecular(0.2)

    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume


# -----------------------------------------------------------#
#                          点云
# -----------------------------------------------------------#
def suv_point_cloud(suv_array: np.ndarray, origin: list, spacing: list, body_array=None):
    shape = suv_array.shape
    suv_image = suvbw2image(suv_array, 2.5, True)

    hot_cmap = matplotlib.colormaps["hot"]
    hot_colors = hot_cmap(np.linspace(0, 1, 256))

    points = vtkPoints()
    lookup = vtkLookupTable()
    scalars = vtkFloatArray()
    cells = vtkCellArray()

    if body_array is None:
        total = np.sum((suv_array > 1e-3).astype(np.uint8))
    else:
        total = np.sum(body_array.astype(np.uint8))
    lookup.SetNumberOfTableValues(total)

    i = 0
    for d in range(shape[0]):
        for h in range(shape[1]):
            for w in range(shape[2]):
                if body_array is None and suv_array[d, h, w] <= 1e-3:
                    continue
                if body_array is not None and body_array[d, h, w] == 0:
                    continue
                scalars.InsertNextTuple1(i)
                x, y, z = np.array([w, h, d]) * np.array(spacing) + np.array(origin)
                points.InsertNextPoint(x, y, z)
                cells.InsertNextCell(1)
                cells.InsertCellPoint(i)
                lookup.SetTableValue(
                    i,
                    *hot_colors[suv_image[d, h, w]][0:3],
                    suv_image[d, h, w] / 255,
                )
                i = i + 1
    lookup.Build()
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(cells)
    polydata.GetPointData().SetScalars(scalars)

    polygonMapper = vtkPolyDataMapper()
    polygonMapper.SetInputData(polydata)
    polygonMapper.SetScalarRange(0, total - 1)
    polygonMapper.SetLookupTable(lookup)
    polygonMapper.Update()

    polygonActor = vtkActor()
    polygonActor.SetMapper(polygonMapper)
    polygonActor.GetProperty().SetPointSize(1)
    polygonActor.GetProperty().SetOpacity(0.35)

    return polygonActor


def main():
    TEXTS = ["infected", "uninfected", "bladder", "lesion"]
    GT_COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    DT_COLOR = [(255, 0, 139), (255, 215, 0), (138, 43, 226), (50, 205, 50)]
    IMG_DIR = os.path.join("Files", "FRI", "image_2mm")

    parser = argparse.ArgumentParser(description="PETCT 三维可视化")
    parser.add_argument("-r", "--result_file", type=str, default="")
    parser.add_argument('-n', "--no", type=str, default="")
    parser.add_argument("--vis", type=str, default="petct")
    parser.add_argument("--no-gt", action="store_true")
    parser.add_argument("--no-dr", action="store_true")
    parser.add_argument("--only-body", action="store_true")
    parser.add_argument("--ct-file", type=str, default=None)
    parser.add_argument("--pet-file", type=str, default=None)
    parser.add_argument("--bbox-file", type=str, default=None)
    args = parser.parse_args()

    # 颜色
    vtk_colors = vtkNamedColors()
    vtk_colors.SetColor('BkgColor', [255, 255, 255, 255])
    # 渲染器
    renderer = vtkRenderer()
    # 渲染窗口
    renderWindow = vtkRenderWindow()
    # 渲染窗口添加渲染器
    renderWindow.AddRenderer(renderer)
    # 交互工具可以开启基于鼠标和键盘的与场景的交互能力
    interactor = vtkRenderWindowInteractor()
    # 交互工具绑定渲染窗口
    interactor.SetRenderWindow(renderWindow)
    interactor.SetInteractorStyle(KeyPressStyle())

    # 读取文件
    if args.no:
        ct_image = sitk.ReadImage(os.path.join(IMG_DIR, f"{args.no}_CT.nii.gz"))
        pet_image = sitk.ReadImage(os.path.join(IMG_DIR, f"{args.no}_SUVbw.nii.gz"))
    elif args.ct_file is None:
        ct_image = sitk.ReadImage(args.ct_file)
    elif args.pet_file is None:
        pet_image = sitk.ReadImage(args.pet_file)

    ct_array = sitk.GetArrayFromImage(ct_image)
    pet_array = sitk.GetArrayFromImage(pet_image)
    ct_array = flipXY(ct_array)
    pet_array = flipXY(pet_array)
    body_array = None
    if args.only_body:
        body_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(IMG_DIR, f"{args.no}_Label_Body.nii.gz")))
        body_array = flipXY(body_array)

    # 渲染器的摄像头
    camera = renderer.GetActiveCamera()
    camera.SetViewUp(0, 0, 1)
    camera.Azimuth(30.0)
    camera.Elevation(30.0)
    spacing = ct_image.GetSpacing()  # 像素间距
    size = ct_image.GetSize()  # 图像大小
    origin = ct_image.GetOrigin()  # 原点
    print(f"spacing: {spacing}, size: {size}, origin: {origin}.")
    whd = np.array(size) * np.array(spacing)  # 长宽高
    focal_point = np.array(origin) + whd / 2
    camera.SetFocalPoint(focal_point)
    position = [
        focal_point[0] + whd[2] * 1.25,
        focal_point[1] + whd[2] * 1.25 * 1.73205,
        focal_point[2],
    ]
    camera.SetPosition(position)

    # 添加坐标轴
    axes = vtkAxesActor()
    transform = vtkTransform()
    transform.Translate(origin)  # 起始点
    transform.Scale(max(size) / 2, max(size) / 2, max(size) / 2)
    axes.SetUserTransform(transform)

    # axes.SetXAxisFontSize(1)
    axes.SetXAxisLabelText("C")
    axes.SetYAxisLabelText("S")
    axes.SetZAxisLabelText("T")
    axes.GetXAxisCaptionActor2D().SetHeight(0.05)
    axes.GetYAxisCaptionActor2D().SetHeight(0.05)
    axes.GetZAxisCaptionActor2D().SetHeight(0.05)
    renderer.AddActor(axes)

    if "ct" in args.vis:
        ct_volume = ct_vtk_volume(ct_array, origin, spacing, body_array)
        renderer.AddViewProp(ct_volume)

    if "pet" in args.vis:
        suv_pc = suv_point_cloud(pet_array, origin, spacing, body_array)
        renderer.AddActor(suv_pc)

    result = json.load(open(args.result_file))
    if not args.no_gt:
        # 读取边界框
        gt_boxes, gt_texts, gt_colors = [], [], []
        labels = result["ground_truth"][args.no]
        for label in labels:
            text, box = label["class_name"], label["bbox"]
            if text not in TEXTS:
                continue
            color = (np.array(GT_COLOR[TEXTS.index(text)]) / 255.0).tolist()
            gt_boxes.append(box)
            gt_texts.append(text)
            gt_colors.append(color)
        gt_boxes = flip_boxXY(gt_boxes, size)
        # 创建渲染的物体 gt 定位框
        gt_actors = vtk_bounding_boxes(gt_boxes, origin, spacing, gt_texts, gt_colors, max(size))
        for i, actor in enumerate(gt_actors):
            renderer.AddActor(actor)
            if i % 2 == 1:
                actor.SetCamera(camera)  # 标签文本跟着摄像头旋转

    if not args.no_dr:
        dr_boxes, dr_texts, dr_colors = [], [], []
        labels = result["detection_results"][args.no]
        for label in labels:
            c, _, x1, y1, z1, x2, y2, z2 = (
                label['class_name'],
                label['confidence'],
                *label['bbox'],
            )
            color = (np.array(DT_COLOR[TEXTS.index(c)]) / 255.0).tolist()
            dr_boxes.append([int(_) for _ in [x1, y1, z1, x2, y2, z2]])
            dr_texts.append(c)
            dr_colors.append(color)
        dr_boxes = flip_boxXY(dr_boxes, size)
        dr_actors = vtk_bounding_boxes(dr_boxes, origin, spacing, dr_texts, dr_colors, max(size), False)
        # 创建渲染的物体 dr 定位框
        for i, actor in enumerate(dr_actors):
            renderer.AddActor(actor)
            if i % 2 == 1:
                actor.SetCamera(camera)  # 标签文本跟着摄像头旋转

    if args.bbox_file is not None:
        boxes, texts, colors = [], [], []
        bbox = json.load(open(args.bbox_file))
        for label in bbox:
            x1, y1, z1, x2, y2, z2, c = label
            color = (np.array(DT_COLOR[c]) / 255.0).tolist()
            boxes.append([x1, y1, z1, x2, y2, z2])
            texts.append(TEXTS[c])
            colors.append(color)
        boxes = flip_boxXY(boxes, size)
        actors = vtk_bounding_boxes(boxes, origin, spacing, texts, colors, max(size))
        # 创建渲染的物体 dr 定位框
        for i, actor in enumerate(actors):
            renderer.AddActor(actor)
            if i % 2 == 1:
                actor.SetCamera(camera)  # 标签文本跟着摄像头旋转

    # 设置背景颜色
    renderer.SetBackground(vtk_colors.GetColor3d('BkgColor'))
    renderer.ResetCameraClippingRange()

    interactor.Initialize()
    renderWindow.SetSize(1920, 1080)
    renderWindow.SetWindowName(f"PETCT Visualization - {args}")
    renderWindow.Render()

    # 开启交互
    interactor.Start()


if __name__ == "__main__":
    main()
