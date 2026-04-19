"""ParaView Python plugin to read fluid_sim frame HDF5 files in xyz order.

Load this file in ParaView via `Tools -> Manage Plugins -> Load New...`.
After that, opening one or more matching `.h5` frame files should offer this
reader. Multiple selected files are exposed as a time series.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from paraview.util.vtkAlgorithm import smdomain, smhint, smproperty, smproxy
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline


@smproxy.reader(
    name="FluidSimHDF5FrameReader",
    label="Fluid Sim HDF5 Frame Reader",
    extensions="h5",
    file_description="Fluid Sim HDF5 Frames",
)
class FluidSimHDF5FrameReader(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkImageData")
        self._filenames: list[str] = []
        self._spacing = (1.0, 1.0, 1.0)
        self._origin = (0.0, 0.0, 0.0)
        self._first_shape: tuple[int, int, int] | None = None

    @smproperty.xml(
        """
        <StringVectorProperty
            name="FileName"
            command="AddFileName"
            clean_command="ClearFileNames"
            animateable="0"
            number_of_elements="1"
            repeat_command="1"
            panel_visibility="never">
          <FileListDomain name="files"/>
          <Hints>
            <FileChooser extensions="h5" file_description="Fluid Sim HDF5 Frames"/>
          </Hints>
        </StringVectorProperty>
        """
    )
    def AddFileName(self, name):
        if name:
            self._filenames.append(str(name))
            self._first_shape = None
            self.Modified()

    def ClearFileNames(self):
        if self._filenames:
            self._filenames = []
            self._first_shape = None
            self.Modified()

    @smproperty.doublevector(name="Spacing", default_values=[1.0, 1.0, 1.0], number_of_elements=3)
    def SetSpacing(self, x, y, z):
        spacing = (float(x), float(y), float(z))
        if spacing != self._spacing:
            self._spacing = spacing
            self.Modified()

    @smproperty.doublevector(name="Origin", default_values=[0.0, 0.0, 0.0], number_of_elements=3)
    def SetOrigin(self, x, y, z):
        origin = (float(x), float(y), float(z))
        if origin != self._origin:
            self._origin = origin
            self.Modified()

    @smproperty.doublevector(name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty")
    def GetTimestepValues(self):
        return list(range(len(self._filenames)))

    def CanReadFile(self, filename):
        try:
            self._read_frame(Path(filename))
            return 1
        except Exception:
            return 0

    def RequestInformation(self, request, inInfoVec, outInfoVec):
        if not self._filenames:
            return 1

        nx, ny, nz = self._get_first_shape()
        executive = self.GetExecutive()
        out_info = outInfoVec.GetInformationObject(0)
        out_info.Set(
            vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT(),
            (0, nx, 0, ny, 0, nz),
            6,
        )

        timesteps = self.GetTimestepValues()
        out_info.Remove(executive.TIME_STEPS())
        out_info.Remove(executive.TIME_RANGE())
        for timestep in timesteps:
            out_info.Append(executive.TIME_STEPS(), timestep)
        if timesteps:
            out_info.Append(executive.TIME_RANGE(), timesteps[0])
            out_info.Append(executive.TIME_RANGE(), timesteps[-1])
        return 1

    def RequestData(self, request, inInfoVec, outInfoVec):
        if not self._filenames:
            raise RuntimeError("No HDF5 frame files selected.")

        file_index = self._resolve_file_index(outInfoVec)
        density_xyz, momentum_xyz = self._read_frame(Path(self._filenames[file_index]))
        nx, ny, nz = density_xyz.shape

        output = vtkImageData.GetData(outInfoVec, 0)
        output.SetOrigin(*self._origin)
        output.SetSpacing(*self._spacing)
        output.SetDimensions(nx + 1, ny + 1, nz + 1)

        density_vtk = numpy_to_vtk(
            np.ascontiguousarray(np.transpose(density_xyz, (2, 1, 0)).reshape(-1)),
            deep=True,
        )
        density_vtk.SetName("density_offset")
        output.GetCellData().AddArray(density_vtk)
        output.GetCellData().SetScalars(density_vtk)

        momentum_vtk = numpy_to_vtk(
            np.ascontiguousarray(np.transpose(momentum_xyz, (2, 1, 0, 3)).reshape(-1, 3)),
            deep=True,
        )
        momentum_vtk.SetName("momentum")
        output.GetCellData().AddArray(momentum_vtk)
        output.GetCellData().SetVectors(momentum_vtk)

        output.GetInformation().Set(output.DATA_TIME_STEP(), float(file_index))
        return 1

    def _get_first_shape(self):
        if self._first_shape is None:
            density_xyz, _ = self._read_frame(Path(self._filenames[0]))
            self._first_shape = tuple(int(axis) for axis in density_xyz.shape)
        return self._first_shape

    def _resolve_file_index(self, outInfoVec):
        executive = self.GetExecutive()
        out_info = outInfoVec.GetInformationObject(0)
        if out_info.Has(executive.UPDATE_TIME_STEP()):
            requested = out_info.Get(executive.UPDATE_TIME_STEP())
            idx = int(round(requested))
            return max(0, min(idx, len(self._filenames) - 1))
        return 0

    def _read_frame(self, path: Path):
        with h5py.File(path, "r") as handle:
            if "density_offset" not in handle or "momentum" not in handle:
                raise RuntimeError(f"{path} is missing density_offset or momentum datasets.")

            density_xyz = np.asarray(handle["density_offset"], dtype=np.float32)
            momentum_xyz = np.asarray(handle["momentum"], dtype=np.float32)

        if density_xyz.ndim != 3:
            raise RuntimeError(f"{path} density_offset must have shape (nx, ny, nz).")
        if momentum_xyz.ndim != 4 or momentum_xyz.shape[-1] != 3:
            raise RuntimeError(f"{path} momentum must have shape (nx, ny, nz, 3).")
        if momentum_xyz.shape[:3] != density_xyz.shape:
            raise RuntimeError(f"{path} momentum and density_offset shapes do not match.")

        return density_xyz, momentum_xyz
