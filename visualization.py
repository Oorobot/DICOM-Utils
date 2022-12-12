import matplotlib as mpl
import numpy as np
import SimpleITK as sitk

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingVolumeOpenGL2
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPiecewiseFunction,
    vtkPolyData,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper

from utils.dicom import HU2image, SUVbw2image
from utils.utils import rot90
from utils.visualization import KeyPressStyle, index_to_physical_position


# 读取 CT 或者 SUVbw 文件
ct_path = "001.nii.gz"
suv_path = "001_.nii.gz"
ct_image = sitk.ReadImage(ct_path)
suv_image = sitk.ReadImage(suv_path)
spacing = ct_image.GetSpacing()

# 定位框的颜色
annotation_colors = {"fraction": "Red", "bladder": "Blue", "Other": "Green"}
# SUVbw color map
hot_cmap = mpl.cm.get_cmap("hot")
hot_colors = hot_cmap(np.linspace(0, 1, hot_cmap.N))

# 自定义原点
origin = (0.0, 0.0, 0.0)

# 图像预处理
ct_array = sitk.GetArrayFromImage(ct_image)
ct_array = HU2image(ct_array, 300.0, 1500.0) * 1500.0

suv_array = sitk.GetArrayFromImage(suv_image)
suv_array[suv_array < 0] = 0
suv_array = SUVbw2image(suv_array, 10.0, True)


boxes = [[41, 80, 72, 78, 102, 96], [82, 70, 113, 111, 97, 117]]
box_colors = ["Red", "Blue"]


# 翻转和旋转
array, b = rot90(ct_array, boxes, 0, (0, 2))
array = sitk.GetArrayFromImage(sitk.GetImageFromArray(array))
physical_points = index_to_physical_position(b, origin, spacing)


# 创建渲染器，渲染窗口和交互工具. 渲染器画入在渲染窗口里，交互工具可以开启基于鼠标和键盘的与场景的交互能力
ren = vtkRenderer()
ren_win = vtkRenderWindow()
ren_win.AddRenderer(ren)
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(ren_win)

ren_win.Render()
iren.SetInteractorStyle(KeyPressStyle())


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

# 添加立方体框
for p, c in zip(physical_points, box_colors):
    points = vtkPoints()
    points.SetNumberOfPoints(8)
    for i in range(8):
        points.SetPoint(i, *p[i])
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
    polygonActor.GetProperty().SetColor(colors.GetColor3d(c))

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
