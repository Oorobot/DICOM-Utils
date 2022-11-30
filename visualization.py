import numpy as np
import SimpleITK as sitk
import vtk
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingVolumeOpenGL2
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (vtkCellArray, vtkPiecewiseFunction,
                                           vtkPolyData)
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (vtkActor, vtkColorTransferFunction,
                                         vtkPolyDataMapper, vtkRenderer,
                                         vtkRenderWindow,
                                         vtkRenderWindowInteractor, vtkVolume,
                                         vtkVolumeProperty)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper

from utils.dicom import ct2image
from utils.utils import load_json


# 设置键盘交互
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
            camera.SetPosition(*new_position.tolist())
        if obj.GetInteractor().GetKeySym() == "s":
            direction_of_projection = camera.GetDirectionOfProjection()
            position = camera.GetPosition()
            new_position = np.array(position) - 100 * np.array(direction_of_projection)
            camera.SetPosition(*new_position.tolist())


def flip_3D_annotation(image_size, point1, point2, axes):
    """
    image_size = (D, H, W): 三维物体的大小
    point1 = (z1, y1, x1): 三维标注中对角的第一个点
    point2 = (z2, y2, x2): 三维标注中对角的第二个点(x2>x1, y2>y1, z2>z1)
    axes: 沿哪个轴进行翻转
    """
    assert axes == 0 or axes == 1 or axes == 2
    point1[axes] = image_size[axes] - point1[axes]
    point2[axes] = image_size[axes] - point2[axes]
    return point1, point2


# 进行翻转和旋转测试
def rot90_3D_annotation(image_size, point1, point2, k, axes):
    """
    image_size = (D, H, W): 三维物体的大小
    point1 = (z1, y1, x1): 三维标注中对角的第一个点
    point2 = (z2, y2, x2): 三维标注中对角的第二个点(x2>x1, y2>y1, z2>z1)
    k: 旋转90度的次数
    axes: 按照哪个平面进行旋转, 在x-y平面, axes = (1, 2) 或者 axes = (2, 1)
    """
    k = k % 4
    y1, x1 = point1[axes[0]], point1[axes[1]]
    y2, x2 = point2[axes[0]], point2[axes[1]]
    H, W = image_size[axes[0]], image_size[axes[1]]
    if k == 0:
        x1_, y1_ = x1, y1
        x2_, y2_ = x2, y2
    elif k == 1:
        x1_, y1_ = y1, W - x2
        x2_, y2_ = y2, W - x1
    elif k == 2:
        x1_, y1_ = W - x2, H - y2
        x2_, y2_ = W - x1, H - y1
    elif k == 3:
        x1_, y1_ = H - y2, x1
        x2_, y2_ = H - y1, x2

    point1[axes[0]], point1[axes[1]] = y1_, x1_
    point2[axes[0]], point2[axes[1]] = y2_, x2_
    return point1, point2


def index_to_physical_position(boxes, origin, spacing):
    physical_positions = []
    origin = np.array(origin)
    spacing = np.array(spacing)
    for box in boxes:
        point1, point2 = np.array(box[0:3]), np.array(box[3:6])
        x1, y1, z1 = origin + point1 * spacing - spacing * 0.5
        x2, y2, z2 = origin + point2 * spacing - spacing * 0.5
        physical_positions.append(
            [
                [x1, y1, z1],
                [x1, y2, z1],
                [x2, y2, z1],
                [x2, y1, z1],
                [x1, y1, z2],
                [x1, y2, z2],
                [x2, y2, z2],
                [x2, y1, z2],
            ]
        )
    return physical_positions


# 读取 CT 或者 SUVbw 文件
ct_path = "Files/resampled_FRI/001_rCT.nii.gz"
suv_path = "Files/resampled_FRI/001_rSUVbw.nii.gz"
ct_image = sitk.ReadImage(ct_path)
suv_image = sitk.ReadImage(suv_path)
print(
    f"CT size: {ct_image.GetSize()}, origin: {ct_image.GetOrigin()}, spacing: {ct_image.GetSpacing()}."
)
print(
    f"SUV size: {suv_image.GetSize()}, origin: {suv_image.GetOrigin()}, spacing: {suv_image.GetSpacing()}."
)

# 自定义原点
origin = (0.0, 0.0, 0.0)

# 图像预处理
ct_array = sitk.GetArrayFromImage(ct_image)
suv_array = sitk.GetArrayFromImage(suv_image)

ct_array = ct2image(ct_array, 300.0, 1500.0) * 1500.0
suv_array[suv_array < 0] = 0


""""""
k = 3
axes = (0, 2)
# array = np.flip(array, 0)
# array = np.flip(array, 2)
shape = array.shape
array = np.rot90(array, k, axes)
array = sitk.GetArrayFromImage(sitk.GetImageFromArray(array))

# 读取标注文件
annotations_path = "./Files/annotations.json"
annotations = load_json(annotations_path)
annotation = annotations["001"]["annotations"]
physical_points = []
for a in annotation:
    point1 = np.array(a["location"][0:3])
    point2 = np.array(a["location"][3:6])
    x1, y1, z1 = origin + point1 * spacing - spacing * 0.5
    x2, y2, z2 = origin + point2 * spacing - spacing * 0.5
    (z1, y1, x1), (z2, y2, x2) = rot90_3D_annotation(
        shape, [z1, y1, x1], [z2, y2, x2], k, axes
    )
    physical_points.append(
        {
            "class": a["class"],
            "points": [
                [x1, y1, z1],
                [x1, y2, z1],
                [x2, y2, z1],
                [x2, y1, z1],
                [x1, y1, z2],
                [x1, y2, z2],
                [x2, y2, z2],
                [x2, y1, z2],
            ],
        }
    )
class_colors = {"fraction": "Red", "bladder": "Blue", "Other": "Green"}
# 创建渲染器，渲染窗口和交互工具. 渲染器画入在渲染窗口里，交互工具可以开启基于鼠标和键盘的与场景的交互能力
ren = vtkRenderer()
ren_win = vtkRenderWindow()
ren_win.AddRenderer(ren)
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(ren_win)

ren_win.Render()
iren.SetInteractorStyle(KeyPressStyle())

# def main():
#     ren1 = vtk.vtkRenderer()
#     renWin = vtk.vtkRenderWindow()
#     renWin.AddRenderer(ren1)
#     iren = vtk.vtkRenderWindowInteractor()
#     iren.SetRenderWindow(renWin)

#     renWin.Render()
#     iren.SetInteractorStyle(vtk.KeyPressStyle())

# 创建 vtk 颜色
colors = vtkNamedColors()

# 创建 vtk 图像
vtk_image = vtkImageImportFromArray()
vtk_image.SetArray(array)
vtk_image.SetDataSpacing(spacing)
vtk_image.SetDataOrigin(origin)
vtk_image.Update()


# 体积将通过光线投射alpha合成显示。
# 需要光线投射映射器来进行光线投射。
volume_mapper = vtkFixedPointVolumeRayCastMapper()
volume_mapper.SetInputConnection(vtk_image.GetOutputPort())

# 使用颜色转换函数，对不同的值设置不同的函数
volume_color = vtkColorTransferFunction()
rgb_point = np.linspace(-450, 1050, 256)
for i in range(0, 256):
    volume_color.AddRGBPoint(rgb_point[i], i / 255.0, i / 255.0, i / 255.0)


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

ren.AddViewProp(volume)

# # 添加坐标轴
# axes = vtkAxesActor()
# # 使用用户变化来定位坐标轴
# transform = vtkTransform()
# transform.Translate(0.0, 0.0, 430.0)
# axes.SetUserTransform(transform)
# ren.AddActor(axes)

# 添加立方体框
for p_p in physical_points:
    points = vtkPoints()
    points.SetNumberOfPoints(8)
    for i in range(8):
        points.SetPoint(i, *p_p["points"][i])
    lines = vtkCellArray()
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

    polygon = vtkPolyData()
    polygon.SetPoints(points)
    polygon.SetLines(lines)

    polygonMapper = vtkPolyDataMapper()
    polygonMapper.SetInputData(polygon)
    polygonMapper.Update()

    polygonActor = vtkActor()
    polygonActor.SetMapper(polygonMapper)
    polygonActor.GetProperty().SetColor(colors.GetColor3d(class_colors[p_p["class"]]))

    ren.AddActor(polygonActor)

# 设置摄像机
camera = ren.GetActiveCamera()
print("volume origin: ", volume.GetOrigin())
print("volume center: ", volume.GetCenter())
# print("volume spacing: ", volume.GetSpacing())
c = volume.GetCenter()
# c = volume.GetOrigin()
camera.SetViewUp(0, 0, -1)
camera.SetPosition(c[0], c[1] - 400, c[2])
camera.SetFocalPoint(c[0], c[1], c[2])
camera.Azimuth(30.0)
camera.Elevation(30.0)

# 设置背景颜色
ren.SetBackground(colors.GetColor3d("White"))

# Increase the size of the render window
ren_win.SetSize(680, 480)
ren_win.SetWindowName("CT Visualization")

# Interact with the data.
iren.Initialize()
iren.Start()
