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
from utils.visualization import (
    KeyPressStyle,
    index_to_physical_position,
    ct_vtk_volume,
    suv_vtk_volume,
    suv_point_cloud,
)


# 读取 CT 或者 SUVbw 文件
ct_path = "001.nii.gz"
suv_path = "001_.nii.gz"
ct_image = sitk.ReadImage(ct_path)
suv_image = sitk.ReadImage(suv_path)
origin = [0.0,0.0,0.0]
spacing = ct_image.GetSpacing()

# 图像预处理
ct_array = sitk.GetArrayFromImage(ct_image)
suv_array = sitk.GetArrayFromImage(suv_image)

# 定位框的颜色
annotation_colors = {"fraction": "Red", "bladder": "Blue", "Other": "Green"}

boxes = [[41, 80, 72, 78, 102, 96], [82, 70, 113, 111, 97, 117]]
box_colors = ["Red", "Blue"]

physical_points = index_to_physical_position(boxes, origin, spacing)


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

ct_volume = ct_vtk_volume(ct_array, [0, 0, 0], spacing)
ren.AddViewProp(ct_volume)
# suv_volume = suv_vtk_volume(suv_array, [0, 0, 0], spacing)
# ren.AddViewProp(suv_volume)
# suv_pc = suv_point_cloud(suv_array,origin,spacing)
# ren.AddActor(suv_pc)
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
# print("volume spacing: ", volume.GetSpacing())
c = (0,0,0)
# c = volume.GetOrigin()
camera.SetViewUp(0, 0, -1)
camera.SetPosition(c[0], c[1] - 400, c[2])
camera.SetFocalPoint(c[0], c[1], c[2])
camera.Azimuth(30.0)
camera.Elevation(30.0)

# 设置背景颜色
ren.SetBackground(colors.GetColor3d("Grey"))

# Increase the size of the render window
ren_win.SetSize(680, 480)
ren_win.SetWindowName("CT Visualization")

# Interact with the data.
iren.Initialize()
iren.Start()
 