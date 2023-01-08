import os

import numpy as np
import SimpleITK as sitk
import vtk

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingVolumeOpenGL2
from matplotlib.cm import get_cmap
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPiecewiseFunction,
    vtkPolyData,
)
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

from utils.dicom import SUVbw2image
from utils.utils import load_json

# -----------------------------------------------------------#
#                       自定义键盘交互
# -----------------------------------------------------------#
class KeyPressStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        super().__init__()
        self.AddObserver("KeyPressEvent", self.OnKeyPress)

    def OnKeyPress(self, obj, event):
        print(f"Key {obj.GetInteractor().GetKeySym()} pressed")
        camera = (
            obj.GetInteractor()
            .GetRenderWindow()
            .GetRenderers()
            .GetFirstRenderer()
            .GetActiveCamera()
        )
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
#                     标注框以及文字
# -----------------------------------------------------------#
def vtk_bounding_boxes(boxes, origin, spacing, texts, colors):
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
        textActor.SetScale(10, 10, 10)
        textActor.AddPosition(x2, y2, z2)
        textActor.GetProperty().SetColor(*color)

        actors.append(textActor)
    return actors


# -----------------------------------------------------------#
#                      vtk volume
# -----------------------------------------------------------#
def ct_vtk_volume(ct_array: np.ndarray, origin: list, spacing: list):
    ct_array = np.clip(ct_array, -450, 1050)
    ct_array = sitk.GetArrayFromImage(
        sitk.GetImageFromArray(ct_array.astype(np.float32))
    )

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
    cmap = get_cmap("gray")
    colors = cmap(np.linspace(0, 1, cmap.N))
    rgb_point = np.linspace(-450, 1050, 256)
    for i in range(0, 256):
        volume_color.AddRGBPoint(rgb_point[i], *colors[i, :3])

    # 使用透明度转换函数，用于控制不同组织之间的透明度
    volume_scalar_opacity = vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.00)
    volume_scalar_opacity.AddPoint(500, 0.15)
    volume_scalar_opacity.AddPoint(1000, 0.15)
    volume_scalar_opacity.AddPoint(1150, 0.85)

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
#                      vtk volume
# -----------------------------------------------------------#
def suv_vtk_volume(suv_array: np.ndarray, origin: list, spacing: list):
    array = SUVbw2image(suv_array, 2.5)
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
    hot_cmap = get_cmap("hot")
    hot_colors = hot_cmap(linspace)
    for i in range(0, 256):
        volume_color.AddRGBPoint(linspace[i], *hot_colors[i, 0:3])

    # 使用透明度转换函数，用于控制不同组织之间的透明度
    volume_scalar_opacity = vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.0)
    volume_scalar_opacity.AddPoint(0.4, 0.15)
    volume_scalar_opacity.AddPoint(0.8, 0.15)
    volume_scalar_opacity.AddPoint(0.9, 0.85)

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
#                     vtk point cloud
# -----------------------------------------------------------#
def suv_point_cloud(suv_array: np.ndarray, origin: list, spacing: list):
    shape = suv_array.shape
    suv_image = SUVbw2image(suv_array, 2.5, True)

    hot_cmap = get_cmap("hot")
    hot_colors = hot_cmap(np.linspace(0, 1, 256))

    points = vtkPoints()
    lookup = vtkLookupTable()
    scalars = vtkFloatArray()
    cells = vtkCellArray()

    total = np.sum((suv_array > 1e-3).astype(np.uint8))
    lookup.SetNumberOfTableValues(total)

    i = 0
    for d in range(shape[0]):
        for h in range(shape[1]):
            for w in range(shape[2]):
                if suv_array[d, h, w] <= 1e-3:
                    continue
                scalars.InsertNextTuple1(i)
                x, y, z = np.array([w, h, d]) * np.array(spacing) + np.array(origin)
                points.InsertNextPoint(x, y, z)
                cells.InsertNextCell(1)
                cells.InsertCellPoint(i)
                lookup.SetTableValue(
                    i, *hot_colors[suv_image[d, h, w]][0:3], suv_image[d, h, w] / 255,
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
    polygonActor.GetProperty().SetOpacity(0.4)

    return polygonActor


if __name__ == "__main__":
    image_id = "447"
    label_json = "Files/image_2mm.json"
    detection_txt = "Files/yolov3/ep600_ap25/detection-results/447.txt"
    input_shape = [384, 96, 160]

    TEXTS = ["infected_lesion", "uninfected_lesion", "bladder"]
    # GT_COLOR = [(16, 161, 157), (84, 3, 117), (255, 112, 0), (255, 191, 0)]
    # DT_COLOR = [(28, 49, 94), (34, 124, 112), (136, 164, 124), (230, 226, 195)]
    GT_COLOR = [(255, 0, 0), (255, 82, 50), (221, 80, 53), (255, 158, 129)]
    DT_COLOR = [(0, 128, 0), (70, 149, 54), (110, 170, 94), (147, 191, 133)]

    # 读取文件
    ct_image = sitk.ReadImage(os.path.join("Files", "2mm", f"{image_id}_CT.nii.gz"))
    suv_image = sitk.ReadImage(os.path.join("Files", "2mm", f"{image_id}_SUVbw.nii.gz"))
    ct_array = sitk.GetArrayFromImage(ct_image)
    suv_array = sitk.GetArrayFromImage(suv_image)

    origin = [0.0, 0.0, 0.0]
    spacing = ct_image.GetSpacing()

    gt_boxes = []
    gt_texts = []
    gt_colors = []
    labels = load_json(label_json)[image_id]["labels"]
    for label in labels:
        text, box = label["category"], label["position"]
        if text not in TEXTS:
            continue
        color = (np.array(GT_COLOR[TEXTS.index(text)]) / 255.0).tolist()
        gt_boxes.append(box)
        gt_texts.append(text)
        gt_colors.append(color)
    lines = open(detection_txt, "r").readlines()
    dt_boxes = []
    dt_texts = []
    dt_colors = []
    for line in lines:
        text, c, *box = line.strip().split()
        color = np.array(DT_COLOR[TEXTS.index(text)]) / 255.0
        dt_texts.append(text)
        dt_colors.append(color)
        dt_boxes.append([int(b) for b in box])

    # 创建需要渲染的物体
    ct_volume = ct_vtk_volume(ct_array, origin, spacing)
    suv_pc = suv_point_cloud(suv_array, origin, spacing)
    gt_actors = vtk_bounding_boxes(gt_boxes, origin, spacing, gt_texts, gt_colors)
    dt_actors = vtk_bounding_boxes(dt_boxes, origin, spacing, dt_texts, dt_colors)

    # 创建渲染器，渲染窗口和交互工具. 渲染器画入在渲染窗口里，交互工具可以开启基于鼠标和键盘的与场景的交互能力
    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)
    interactor.SetInteractorStyle(KeyPressStyle())

    renderer.AddViewProp(ct_volume)
    renderer.AddActor(suv_pc)
    for actor in gt_actors:
        renderer.AddActor(actor)
    for actor in dt_actors:
        renderer.AddActor(actor)

    # 设置摄像机
    camera = renderer.GetActiveCamera()
    c = ct_volume.GetCenter()
    camera.SetViewUp(0, 0, -1)
    camera.SetPosition(c[0], c[1] - 400, c[2])
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.Azimuth(30.0)
    camera.Elevation(30.0)

    # 设置背景颜色
    renderer.SetBackground(vtkNamedColors().GetColor3d("Grey"))

    renderer.ResetCameraClippingRange()
    for i in range(1, len(gt_actors), 2):
        gt_actors[i].SetCamera(renderer.GetActiveCamera())
    for i in range(1, len(dt_actors), 2):
        dt_actors[i].SetCamera(renderer.GetActiveCamera())

    interactor.Initialize()

    renderWindow.SetSize(680, 480)
    renderWindow.SetWindowName(f"PETCT Visualization - {image_id}")
    renderWindow.Render()

    # 开启交互
    interactor.Start()
