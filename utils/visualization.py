import numpy as np
import SimpleITK as sitk
import vtk
from matplotlib.cm import get_cmap
from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkFloatArray, vtkLookupTable
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPiecewiseFunction,
    vtkPolyData,
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkColorTransferFunction,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper

from utils.dicom import HU2image, SUVbw2image


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
            camera.SetPosition(*new_position.tolist())
        if obj.GetInteractor().GetKeySym() == "s":
            direction_of_projection = camera.GetDirectionOfProjection()
            position = camera.GetPosition()
            new_position = np.array(position) - 100 * np.array(direction_of_projection)
            camera.SetPosition(*new_position.tolist())


# -----------------------------------------------------------#
#                   索引号 -. 实际坐标位置
# -----------------------------------------------------------#
def index_to_physical_position(boxes, origin, spacing):
    physical_positions = []
    origin = np.array(origin)
    spacing = np.array(spacing)
    for box in boxes:
        point1, point2 = np.array(box[0:3]), np.array(box[3:6])
        x1, y1, z1 = origin + point1 * spacing
        x2, y2, z2 = origin + point2 * spacing
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
    cmap = get_cmap("bone")
    colors = cmap(np.linspace(0, 1, cmap.N))
    rgb_point = np.linspace(-450, 1050, 256)
    for i in range(0, 256):
        volume_color.AddRGBPoint(rgb_point[i], *colors[i, :3])

    # 使用透明度转换函数，用于控制不同组织之间的透明度
    volume_scalar_opacity = vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(0, 0.00)
    volume_scalar_opacity.AddPoint(500, 0.20)
    volume_scalar_opacity.AddPoint(1000, 0.50)
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
                    i,
                    *hot_colors[suv_image[d, h, w]],
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
