from dataclasses import dataclass, field, asdict
from pathlib import Path
import csv
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getEncode(filepath: str) -> str:
    encodings = ["iso-2022-jp", "euc-jp", "shift_jis", "utf-8"]
    for encoding in encodings:
        try:
            with open(filepath, encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not determine encoding for {filepath}")


@dataclass
class MetaData:
    file_name: str
    fileType: Optional[str] = None
    deadTime: Optional[float] = None
    countingTime: Optional[float] = None
    powerNumber: Optional[float] = None
    anodeVoltage: Optional[float] = None
    step: Optional[float] = None
    model: Optional[str] = None
    yAxisMaximum: Optional[float] = None
    startEnergy: Optional[float] = None
    finishEnergy: Optional[float] = None
    flagDifDataGroundLevel: Optional[int] = None
    bgCountingRate: Optional[float] = None
    measureDate: Optional[str] = None
    sampleName: Optional[str] = None
    uvIntensity59: Optional[float] = None
    targetUv: Optional[float] = None
    nameLightCorrection: Optional[str] = None
    sensitivity1: Optional[float] = None
    sensitivity2: Optional[float] = None
    # estimation outputs
    thresholdEnergy: Optional[float] = None
    slope: Optional[float] = None
    yslice: Optional[float] = None
    bg: Optional[float] = None


@dataclass
class CalcData:
    uvEnergy: np.ndarray = field(default_factory=lambda: np.array([]))
    countingRate: np.ndarray = field(default_factory=lambda: np.array([]))
    countingCorrection: np.ndarray = field(default_factory=lambda: np.array([]))
    photonCorrection: np.ndarray = field(default_factory=lambda: np.array([]))
    uvIntensity: np.ndarray = field(default_factory=lambda: np.array([]))
    flGrandLevel: np.ndarray = field(default_factory=lambda: np.array([]))
    flRegLevel: np.ndarray = field(default_factory=lambda: np.array([]))
    pyield: np.ndarray = field(default_factory=lambda: np.array([]))
    ydata: np.ndarray = field(default_factory=lambda: np.array([]))
    npyield: np.ndarray = field(default_factory=lambda: np.array([]))
    nayield: np.ndarray = field(default_factory=lambda: np.array([]))
    guideline: np.ndarray = field(default_factory=lambda: np.array([]))


class AcConv:
    """AC data converter with dataclass-backed metadata and calculation data.

    This class keeps the same public API and attributes as the original code so
    external callers using `uvEnergy`, `ydata`, etc. continue to work.
    """

    def __init__(self, file_name: str):
        self.file_name = Path(file_name)
        # dataclass containers
        self.metadata: Optional[MetaData] = None
        self.calcdata: Optional[CalcData] = None
        # --- keep direct attributes for backward compatibility ---
        # these will be set by _read_para() and calculations
        self.uvEnergy = np.array([])
        self.countingRate = np.array([])
        self.countingCorrection = np.array([])
        self.photonCorrection = np.array([])
        self.uvIntensity = np.array([])
        self.flGrandLevel = np.array([])
        self.flRegLevel = np.array([])
        self.ydata = np.array([])
        self.pyield = np.array([])
        self.npyield = np.array([])
        self.nayield = np.array([])
        self.guideline = np.array([])
        # scalar metadata placeholders
        self.fileType = None
        self.deadTime = None
        self.countingTime = None
        self.powerNumber = None
        self.anodeVoltage = None
        self.step = None
        self.model = None
        self.yAxisMaximum = None
        self.startEnergy = None
        self.finishEnergy = None
        self.flagDifDataGroundLevel = None
        self.bgCountingRate = None
        self.measureDate = None
        self.sampleName = None
        self.uvIntensity59 = None
        self.targetUv = None
        self.nameLightCorrection = None
        self.sensitivity1 = None
        self.sensitivity2 = None
        # free-form storage for JSON
        self.json = None
        self.df = None

    # ---------------- file parsing -----------------
    def _read_para(self) -> None:
        enc = getEncode(str(self.file_name))
        with open(str(self.file_name), encoding=enc) as f:
            reader = csv.reader(f)
            meta = [row for row in reader]

        if len(meta[0]) == 10:
            meta[0].extend(["0", "0.0"])
            meta[2].extend(["1", "1"])

        # map to attributes
        self.fileType = meta[0][0]
        self.deadTime = float(meta[0][1])
        self.countingTime = float(meta[0][2])
        self.powerNumber = float(meta[0][3])
        self.anodeVoltage = float(meta[0][4])
        self.step = float(meta[0][5])
        self.model = meta[0][6]
        self.yAxisMaximum = float(meta[0][7])
        self.startEnergy = float(meta[0][8])
        self.finishEnergy = float(meta[0][9])
        self.flagDifDataGroundLevel = int(meta[0][10])
        self.bgCountingRate = float(meta[0][11])
        self.measureDate = meta[1][0]
        self.sampleName = meta[1][1]
        self.uvIntensity59 = float(meta[2][0])
        self.targetUv = float(meta[2][1])
        self.nameLightCorrection = meta[2][2]
        self.sensitivity1 = float(meta[2][3])
        self.sensitivity2 = float(meta[2][4])

        raw_data = [list(x) for x in zip(*meta[3:])]
        self.uvEnergy = np.array([float(v) for v in raw_data[0]])
        self.countingRate = np.array([float(v) for v in raw_data[1]])
        self.flGrandLevel = np.array([int(v) for v in raw_data[2]])
        self.flRegLevel = np.array([int(v) for v in raw_data[3]])
        self.uvIntensity = np.array([float(v) for v in raw_data[4]])

        # reflect into dataclasses
        self._sync_dataclasses_from_attrs()

    # -------------- calibrations and derived data ----------------
    def _count_calibration(self) -> np.ndarray:
        """Dead-time and background correction of the raw counting rate.

        part1: dead-time correction of the sample counting rate.
        part2: sensitivity/nonlinearity correction (0.13571, 0.0028 are
            manufacturer-supplied calibration constants for the RIKEN KEIKI
            AC-series counting circuit; not user-tunable, see ref. [7] in the
            associated paper).
        part3/part4: the same two corrections applied to the background
            counting rate, subtracted to remove the instrument background.
        """
        if self.model in ("AC-3", "AC-2"):
            self.countingCorrection = self.countingRate
        else:
            part1 = (self.countingRate) / (1 - self.deadTime * (self.countingRate))
            part2 = np.exp(0.13571 / (1 - 0.0028 * (self.countingRate))) * self.sensitivity1
            part3 = (self.bgCountingRate) / (1 - self.deadTime * (self.bgCountingRate))
            part4 = np.exp(0.13571 / (1 - 0.0028 * (self.bgCountingRate))) * self.sensitivity1
            self.countingCorrection = part1 * part2 - part3 * part4
        self._sync_dataclasses_from_attrs()
        return self.countingCorrection

    def _photon_calibration(self) -> np.ndarray:
        """Normalize the counting rate by the incident photon flux.

        0.625 converts the monitor's UV intensity reading to a photon-count
        basis, and 5.9 eV is the reference photon energy at which the
        instrument's UV-intensity calibration (uvIntensity59) is defined;
        both are fixed RIKEN KEIKI AC-series instrument constants (see ref.
        [7] in the associated paper), not user-tunable parameters.
        """
        self.nPhoton = 0.625 * (self.uvIntensity / self.uvEnergy)
        self.unitPhoton = (self.uvIntensity59 * 0.625) / 5.9
        self.photonCorrection = self.nPhoton / self.unitPhoton
        self._sync_dataclasses_from_attrs()
        return self.photonCorrection

    def _pyield_intensity(self) -> Tuple[np.ndarray, np.ndarray]:
        self.ydata = self.countingCorrection / self.photonCorrection
        self.ydata = np.where(self.ydata < 0, 0, self.ydata)
        self.npyield = np.power(self.ydata, self.powerNumber)
        self.npyield[np.isnan(self.npyield)] = 0
        self._sync_dataclasses_from_attrs()
        return self.ydata, self.npyield

    # ------------------- user fit / estimation -------------------
    @staticmethod
    def relu(xdata: np.ndarray, a: float, b: float, bg: float) -> np.ndarray:
        ip = (bg - b) / a
        u = (xdata - ip)
        return a * u * (u > 0.0) + bg

    @staticmethod
    def user_fit(bg_ydata: np.ndarray, reg_xdata: np.ndarray, reg_ydata: np.ndarray, printf: bool = False) -> dict:
        bg = np.nanmean(bg_ydata)
        popt = np.polyfit(reg_xdata, reg_ydata, 1)
        a = popt[0]
        b = popt[1]
        cross_point = (bg - b) / a
        if printf:
            print(f"bg:{bg}, a(slope):{a}, b(yslice):{b}")
            print(f"thresholdEnergy -> {cross_point}")
        return {"thresholdEnergy": cross_point, "slope": a, "yslice": b, "bg": bg}

    def user_estimation(self) -> None:
        self.bg_flag_ind = np.where(self.flGrandLevel == -1)[0].tolist()
        self.reg_flag_ind = np.where(self.flRegLevel == -1)[0].tolist()

        if self.bg_flag_ind != [] and self.reg_flag_ind != []:
            if self.flagDifDataGroundLevel == -1:
                bg_ave = np.nanmean(self.ydata[self.bg_flag_ind])
                c_pys = self.ydata - bg_ave
                self.cc_pys = np.where(c_pys < 0, 0, c_pys)
                self.cc_npys = np.power(self.cc_pys, self.powerNumber)

                bg_ydata = np.array([0.0] * len(self.bg_flag_ind))
                reg_xdata = self.uvEnergy[self.reg_flag_ind]
                reg_ydata = self.cc_npys[self.reg_flag_ind]
            else:
                self.cc_pys = self.ydata
                self.cc_npys = np.power(self.cc_pys, self.powerNumber)
                reg_xdata = self.uvEnergy[self.reg_flag_ind]
                reg_ydata = self.cc_npys[self.reg_flag_ind]
                bg_ydata = self.cc_npys[self.bg_flag_ind]

            self.estimate_value = AcConv.user_fit(bg_ydata, reg_xdata, reg_ydata, printf=False)
            self.nayield = self.cc_npys
            self.guideline = AcConv.relu(
                xdata=self.uvEnergy,
                a=self.estimate_value["slope"],
                b=self.estimate_value["yslice"],
                bg=self.estimate_value["bg"],
            )
        else:
            self.estimate_value = {"thresholdEnergy": np.nan, "slope": np.nan, "yslice": np.nan, "bg": np.nan}
            self.nayield = self.npyield
            self.guideline = np.array([np.nan] * len(self.uvEnergy.tolist()))

        # reflect estimates into metadata dataclass if present
        if self.metadata is None:
            self.metadata = MetaData(file_name=self.file_name.name)
        self.metadata.thresholdEnergy = float(self.estimate_value.get("thresholdEnergy", np.nan))
        self.metadata.slope = float(self.estimate_value.get("slope", np.nan))
        self.metadata.yslice = float(self.estimate_value.get("yslice", np.nan))
        self.metadata.bg = float(self.estimate_value.get("bg", np.nan))
        self._sync_dataclasses_from_attrs()

    # ------------------- dataclass sync helpers -------------------
    def _sync_dataclasses_from_attrs(self) -> None:
        """Populate (or update) the metadata and calcdata dataclasses from
        the plain attributes, keeping attribute names available for backwards
        compatibility.
        """
        if self.metadata is None:
            self.metadata = MetaData(file_name=self.file_name.name)
        # copy scalar metadata fields
        for k in [
            "fileType",
            "deadTime",
            "countingTime",
            "powerNumber",
            "anodeVoltage",
            "step",
            "model",
            "yAxisMaximum",
            "startEnergy",
            "finishEnergy",
            "flagDifDataGroundLevel",
            "bgCountingRate",
            "measureDate",
            "sampleName",
            "uvIntensity59",
            "targetUv",
            "nameLightCorrection",
            "sensitivity1",
            "sensitivity2",
        ]:
            if hasattr(self, k):
                setattr(self.metadata, k, getattr(self, k))

        # calcdata
        self.calcdata = CalcData(
            uvEnergy=self.uvEnergy,
            countingRate=self.countingRate,
            countingCorrection=self.countingCorrection,
            photonCorrection=self.photonCorrection,
            uvIntensity=self.uvIntensity,
            flGrandLevel=self.flGrandLevel,
            flRegLevel=self.flRegLevel,
            ydata=self.ydata,
            pyield=self.ydata,
            npyield=self.npyield,
            nayield=self.nayield,
            guideline=self.guideline,
        )

    # ------------------- export helpers -------------------
    def _make_metadata(self) -> None:
        # Build JSON and DataFrame using dataclasses
        if self.metadata is None or self.calcdata is None:
            self._sync_dataclasses_from_attrs()

        meta_dict = asdict(self.metadata)
        calc_dict = {k: v.tolist() for k, v in asdict(self.calcdata).items()}
        merged = {**meta_dict, **calc_dict}
        self.json = json.dumps(merged, indent=4)
        self.df = pd.DataFrame({
            "uvEnergy": self.calcdata.uvEnergy,
            "countingCorrection": self.calcdata.countingCorrection,
            "photonCorrection": self.calcdata.photonCorrection,
            "pyield": self.calcdata.ydata,
            "ydata": self.calcdata.ydata,
            "npyield": self.calcdata.npyield,
        })

    def export_df2csv(self, df_out_file_name: str = None) -> None:
        if df_out_file_name is None:
            df_out_file_name = self.file_name.with_suffix(".csv")
        if self.df is None:
            self._make_metadata()
        self.df.to_csv(df_out_file_name, index=False)

    def export_json(self, json_out_file_name: str = None) -> None:
        if json_out_file_name is None:
            json_out_file_name = self.file_name.with_suffix(".json")
        if self.json is None:
            self._make_metadata()
        with open(json_out_file_name, "w") as f:
            f.write(self.json)

    # ------------------- plotting -------------------
    def plot_ax(self, axi=None):
        if axi is None:
            fig_ = plt.figure()
            ax_ = fig_.add_subplot(111)
        else:
            ax_ = axi

        title = self.metadata.sampleName if self.metadata and self.metadata.sampleName else self.file_name.name
        ax_.set_title(f"{title}")
        ax_.plot(self.uvEnergy, self.npyield, "ro", label="Data")

        thr = self.metadata.thresholdEnergy if self.metadata else np.nan
        if thr is not None and not np.isnan(thr):
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            ax_.plot(self.uvEnergy, self.guideline, linestyle="-", label=f"User Threshold: {self.metadata.thresholdEnergy:.2f}eV Slope:{self.metadata.slope:.2f}")
            ax_.axvline(self.metadata.thresholdEnergy)
            ax_.text(self.metadata.thresholdEnergy, np.nanmax(self.npyield) * 0.3, f"{self.metadata.thresholdEnergy:.2f}")

        ax_.set_xlabel("Energy [eV]")
        ax_.legend()
        ax_.grid()

        if axi is None:
            plt.show()
            return
        return ax_

    # ------------------- main convert -------------------
    def convert(self) -> None:
        self._read_para()
        self.countingCorrection = self._count_calibration()
        self.photonCorrection = self._photon_calibration()
        self.ydata, self.npyield = self._pyield_intensity()
        self.user_estimation()
        self._make_metadata()



class AdvAcConv(AcConv):
    """Advanced converter that trims arrays when limits are exceeded."""
    
    limit_x_value: float = 6.81
    limit_count_ac2_ac3: int = 2000
    limit_count_other: int = 4000

    def __init__(self, file_name: str):
        super().__init__(file_name)

    def convert(self) -> None:
        self._read_para()
        self.countingCorrection = self._count_calibration()
        self.photonCorrection = self._photon_calibration()
        self.ydata, self.npyield = self._pyield_intensity()
        self.user_estimation()
        self._trim_array_energy()
        self._trim_array_maxcount()
        self._make_metadata()

    def _apply_trim(self, cut_index: int) -> None:
        max_len = len(self.uvEnergy)
        cut_index = max(0, min(int(cut_index), max_len))
        for a in [
            "uvEnergy",
            "countingCorrection",
            "countingRate",
            "photonCorrection",
            "flGrandLevel",
            "flRegLevel",
            "uvIntensity",
            "ydata",
            "pyield",
            "npyield",
            "nayield",
            "guideline",
        ]:
            if hasattr(self, a):
                val = getattr(self, a)
                setattr(self, a, val[:cut_index])
        # keep dataclasses in sync
        self._sync_dataclasses_from_attrs()

    def _trim_array_energy(self) -> None:
        
        condition = self.uvEnergy >= self.limit_x_value
        if np.any(condition):
            trim_index = int(np.argmax(condition))
        else:
            trim_index = len(self.uvEnergy)
        self._apply_trim(trim_index)

    def _trim_array_maxcount(self) -> None:
        if self.model in ("AC-3", "AC-2"):
            limit_count = self.limit_count_ac2_ac3
        else:
            limit_count = self.limit_count_other
            
        condition = self.countingCorrection >= limit_count
        if np.any(condition):
            trim_index = int(np.argmax(condition))
        else:
            trim_index = len(self.countingCorrection)
        self._apply_trim(trim_index)



if __name__ == "__main__":
  
    # Process with default threshold (6.81)
    # adv_converter1 = AdvAcConv("your_file.dat")
    # adv_converter1.convert()
    # print(f"Default limit_x_value: {adv_converter1.limit_x_value}")

    # Process by changing the class-level threshold
    # AdvAcConv.limit_x_value = 7.0
    # adv_converter2 = AdvAcConv("your_file.dat")
    # adv_converter2.convert()
    # print(f"Class-level changed limit_x_value: {adv_converter2.limit_x_value}")

    # Process by changing the instance-level threshold
    # adv_converter3 = AdvAcConv("your_file.dat")
    # adv_converter3.limit_x_value = 6.5
    # adv_converter3.convert()
    # print(f"Instance-level changed limit_x_value: {adv_converter3.limit_x_value}")

    # The default limit_x_value remains 7.0
    # adv_converter4 = AdvAcConv("your_file.dat")
    # print(f"Default limit_x_value is still: {adv_converter4.limit_x_value}")
    pass
