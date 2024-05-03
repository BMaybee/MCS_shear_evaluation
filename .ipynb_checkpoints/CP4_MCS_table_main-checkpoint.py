import MCS_table_create
import numpy as np
import xarray as xr
import datetime
import glob
import pandas as pd
import argparse
import time
import warnings
# Important: CP4 SIMULATION USES CF-TIME CONVENTIONS + 360 DAY CALENDAR
import cftime
import metpy.calc as mpcalc
from metpy.units import units
warnings.filterwarnings("ignore")

# Routine to covert TOA OLR field to brightness temperature Tb
def olr_to_bt(olr):
    #Application of Stefan-Boltzmann law
    sigma = 5.670373e-8
    tf = (olr/sigma)**0.25
    #Convert from bb to empirical BT (degC) - Yang and Slingo, 2001
    a = 1.228
    b = -1.106e-3
    Tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return Tb - 273.15

def load_file(ffile,var):
    try:
        ds=xr.open_dataset(ffile)[var]
    except:
        #Deal with funny landseamask file
        ds=xr.open_dataset(ffile,decode_times=False)[var]
        ds=ds.rename({"rlat":"latitude","rlon":"longitude"})
    ds=ds.assign_coords({"longitude":(ds.longitude-360)})

    try:
        ds=ds.isel(pressure=slice(None,None,-1))
    except:
        pass

    # Note region here is a global variable. Sahel region domain from Senior et al, 2021
    if region == "sahel":
        ds=ds.sel(latitude=slice(9,19),longitude=slice(-12,12))
    # Following regions align with Barton et al global MCS + soil moisture analysis
    if region == "wafrica":
        ds=ds.sel(latitude=slice(4,25),longitude=slice(-18,25))
    if region == "safrica":
        ds=ds.sel(latitude=slice(-35,-15),longitude=slice(20,35))
    return ds

#######################################################################################################
# Primary function for identifying MCS snapshots and building table. Res is approximate model resolution, used to determine pixel threshold cf 5000km2
# Function covers one timestep only; table rows in basic_tab/tab objects.
#######################################################################################################
def make_table(date,hour,region,res=4.5):
    if region == "sahel" or region =="wafrica":
        utc_offset = 0   
    if region == "safrica":
        utc_offset = -3

    hour = hour + utc_offset
    # CP4 files names contain day and next day.
    edate_str="%04d%02d%02d" % (date.year,date.month,date.day)
    ndate = date + datetime.timedelta(days=1)
    ndate_str="%04d%02d%02d" % (ndate.year,ndate.month,ndate.day)
    
    
    # STASH a03332 = "TOA outgoing LW after BL" (W/m2)
    olr_file=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/a03332/a03332_A1hr_mean_?????_4km_{}0030-{}2330.nc".format(edate_str,edate_str))[0]
    # Note use of isel - files span 0-23UTC
    # Timesteps here are on half-hours
    olr=load_file(olr_file,"a03332").isel(time=hour)
    olr = olr_to_bt(olr) # convert olr to bt
    # Mask out ocean areas - only want continental MCSs
    mask = load_file('/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/ANCILS/landseamask_ancil_4km.nc','lsm')
    mask = mask[0,0,:,:].interp(latitude=olr.latitude,longitude=olr.longitude)
    olr = olr.where(mask>0)
    # Identification of MCSs from BT field. Constraints are Tb <= -50C, area >= 5000/res**2
    basic_tab = MCS_table_create.process_tir_image(olr, res)

    # Sample rainfall records for MCSs at detection time. STASH b04203 = "large-scale rainfall"
    # Rainfall sampling at 0.1deg, averaging over spurious high native grid explicit rainfall values
    precip_file=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/a04203/a04203_A1hr_mean_?????_4km_{}0030-{}2330.nc".format(edate_str,edate_str))[0]
    # Native data in kg/m2; convert to mm/h
    precip=3600*load_file(precip_file,"a04203")
    tab = MCS_table_create.add_environment_toTable(precip,basic_tab,date,envvar_take=[],rainvar=True)

    ## Population of 12UTC MCS environments from native grid fields (multiple grids), sampled at 0.7deg.###
    # Get surface pressure field to mask out sub-surface values (0 in model output, must be removed)
    # STASH c00409 = surface air pressure (Pa)
    pfile=glob.glob('/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/c00409/c00409_A1hr_inst_?????_4km_{}0100-{}0000.nc'.format(edate_str,ndate_str))[0]
    env=load_file(pfile,"c00409")/100.
    tab=MCS_table_create.add_environment_toTable(env,tab,date,envvar_take=["psfc"],env_hour=12+utc_offset)
    # Isolate environmental time for masking fields.
    psfc=env[env.time.dt.hour==12+utc_offset]
    
    #BL environment: load variables for 12LT same day. q, twcw zonal shear, surface p and T.
    # u_wind STASH 30201; specific q 30205; tcw not available as single field, need to calculate
    prof_u=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/f30201/f30201_A3hr_inst_?????_4km_{}0300-{}0000.nc".format(edate_str,ndate_str))[0] # units m/s
    prof_q=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/f30205/f30205_A3hr_inst_?????_4km_{}0300-{}0000.nc".format(edate_str,ndate_str))[0] # units kg/kg
    tcol_wetm=glob.glob('/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/c30404/c30404_A1hr_inst_?????_4km_{}0100-{}0000.nc'.format(edate_str,ndate_str))[0] # units kg/m2
    tcol_drym=glob.glob('/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/c30403/c30403_A1hr_inst_?????_4km_{}0100-{}0000.nc'.format(edate_str,ndate_str))[0] # units kg/m2
    tcw=load_file(tcol_wetm,"c30404")-load_file(tcol_drym,"c30403") # units mm (kg/m2 / 1000kg/m2 * 1000 -> mm)

    # Profile fields contain 0's for sub-surface values, which can erroneously reduce field values.
    # Thus must apply surface pressure mask!
    env_u=load_file(prof_u,"f30201")
    env_u=env_u[env_u.time.dt.hour==12+utc_offset]
    env_q=load_file(prof_q,"f30205")
    env_q=env_q[env_q.time.dt.hour==12+utc_offset]
    # Surface pressure on different native UM grid; u and q on same grid.
    psfc=psfc.interp_like(env_q)

    # Apply pressure masking
    env_u650=env_u.sel(pressure=650)
    env_u850=env_u.sel(pressure=850).where(psfc>850)
    env_u925=env_u.sel(pressure=925).where(psfc>925)
    env_q850=env_q.sel(pressure=850).where(psfc>850)
    env_q925=env_q.sel(pressure=925).where(psfc>925)

    # Routines to sample environmental fields for MCSs detected in table
    tab = MCS_table_create.add_environment_toTable(tcw,tab,date,envvar_take=["tcw"],env_hour=12)
    tab = MCS_table_create.add_environment_toTable(env_q925,tab,date,envvar_take=['q925'],env_hour=12+utc_offset)
    tab = MCS_table_create.add_environment_toTable(env_q850,tab,date,envvar_take=['q850'],env_hour=12+utc_offset)
    tab = MCS_table_create.add_environment_toTable(env_u650-env_u925,tab,date,envvar_take=['ushear650_925'],env_hour=12+utc_offset)
    tab = MCS_table_create.add_environment_toTable(env_u650-env_u850,tab,date,envvar_take=['ushear650_850'],env_hour=12+utc_offset)

    # Only keep tabulated data.
    tab.pop('cloudMask')
    tab.pop('tir')
    return pd.DataFrame(tab)


#######################################################################################################
# Available regions: "sahel", "wafrica", "safrica"
# Available years: 1997 - 2006 inclusive
# Important note: to facilitate parallelisation code only handles one year of CP4 data. Then run array job on batch server.
#######################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--region", required=False, type=str, default="wafrica")
parser.add_argument("-y", "--year", required=False, type=int, default=2006)
args = parser.parse_args()

# NOTE: ALL PATHS USED AVAILABLE TO JASMIN impala gws USERS (contact: Elizabeth Kendon)

region=args.region.lower()
year=args.year
# To capture peak MCS activity, for S. Hemisphere want winter DYMAMOND period (slightly tweaked to ensure 40 days from 360 day calendar)
if region=="safrica":
    period=xr.cftime_range(start="{}-01-20".format(year),end="{}-02-29".format(year),calendar="360_day")
    utc_offset=-3
else:
    period=xr.cftime_range(start="{}-08-01".format(year),end="{}-09-10".format(year),calendar="360_day")
    utc_offset=0

mcs_records=[]
for date in period:
    print(date)
    # Sample MCSs hourly between 16 and 22UTC (inclusive)
    for hour in np.arange(16,23):
        mcs_records.append(make_table(date,hour,region))

mcs_records=pd.concat(mcs_records)
outfile = 'CP4_{}_{}_MCS_table'.format(year,region.capitalize())
mcs_records.to_csv(outfile,index=False)