import matplotlib as mpl
import numpy as np
import SimpleITK as sitk
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingVolumeOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (vtkRenderer, vtkRenderWindow,
                                         vtkRenderWindowInteractor)

from utils.visualization import (KeyPressStyle, ct_vtk_volume,
                                 suv_point_cloud,
                                 vtk_bounding_boxes)

# 读取 CT 或者 SUVbw 文件
ct_path = "001.nii.gz"
suv_path = "001_.nii.gz"
ct_image = sitk.ReadImage(ct_path)
suv_image = sitk.ReadImage(suv_path)
ct_array = sitk.GetArrayFromImage(ct_image)
suv_array = sitk.GetArrayFromImage(suv_image)

origin = [.0,.0,.0]
spacing = ct_image.GetSpacing()
boxes = [[41, 80, 72, 78, 102, 96,0], [82, 70, 113, 111, 97, 117,2]]

# 创建需要渲染的物体
ct_volume = ct_vtk_volume(ct_array, origin, spacing)
suv_pc = suv_point_cloud(suv_array,origin,spacing)
actors = vtk_bounding_boxes(boxes,origin,spacing)

# 创建渲染器，渲染窗口和交互工具. 渲染器画入在渲染窗口里，交互工具可以开启基于鼠标和键盘的与场景的交互能力
renderer = vtkRenderer()
renderWindow = vtkRenderWindow()
renderWindow.AddRenderer(renderer)
interactor = vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)
interactor.SetInteractorStyle(KeyPressStyle())

renderer.AddViewProp(ct_volume)
renderer.AddActor(suv_pc)
for actor in actors:
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
for i in range(1,len(actors),2):
    actors[i].SetCamera(renderer.GetActiveCamera())

interactor.Initialize()

renderWindow.SetSize(680, 480)
renderWindow.SetWindowName("PETCT Visualization")
renderWindow.Render()

# 开启交互
interactor.Start()
 