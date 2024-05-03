import MCS_table_create
import numpy as np
import xarray as xr
import datetime
import glob
import pandas as pd
import pickle
import argparse
import time
import warnings
import cftime
from scipy.ndimage import label
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

    #Note region here is a global variable. Sahel region domain from Senior et al, 2021
    if region == "sahel":
        ds=ds.sel(latitude=slice(9,19),longitude=slice(-12,12))
    # Following regions align with Barton et al global MCS + soil moisture analysis
    if region == "wafrica":
        ds=ds.sel(latitude=slice(4,25),longitude=slice(-18,25))
    if region == "safrica":
        ds=ds.sel(latitude=slice(-35,-15),longitude=slice(20,35))
    return ds

#######################################################################################################
# Routine to build full depth atmospheric thermoydnamic profiles
# Profile constructed for calendar day between environmental time and latest MCS-time profile field
# Profiles built from pressure level data but include surface fields
#######################################################################################################
def build_native_profile(date,iflw=True,level="all"):
    t1=date+pd.Timedelta(12+utc_offset,"h")
    t2=date+pd.Timedelta(21+utc_offset,"h")
    # CP4 files names contain day and next day.
    edate_str="%04d%02d%02d" % (date.year,date.month,date.day)
    ndate = date + datetime.timedelta(days=1)
    ndate_str="%04d%02d%02d" % (ndate.year,ndate.month,ndate.day)

    # Surface file paths
    psfc_file=glob.glob('/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/c00409/c00409_A1hr_inst_?????_4km_{}0100-{}0000.nc'.format(edate_str,ndate_str))[0]
    tsfc_file=glob.glob('/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/c03236/c03236_A1hr_inst_?????_4km_{}0100-{}0000.nc'.format(edate_str,ndate_str))[0] 
    qsfc_file=glob.glob('/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/c03237/c03237_A1hr_inst_?????_4km_{}0100-{}0000.nc'.format(edate_str,ndate_str))[0]
    tolw_file=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/a03332/a03332_A1hr_mean_?????_4km_{}0030-{}2330.nc".format(edate_str,edate_str))[0]

    # Variables to include in profile field:
    #     - t : air temperature (K)
    #     - td : dewpoint temperature (K)
    #     - q : specific humidity (g/kg)
    #     - te : equivalent potential temperature (K)
    #     - p : air pressure (hPa)
    #     - toaolr : TOA OLR (2D field; W/m2)
    sfc_t=load_file(tsfc_file,"c03236").rename("air_temperature")
    sfc_t=sfc_t.sel(time=slice(t1,t2,3)) # keep 3 hourly data only to match profiles
    sfc_q=load_file(qsfc_file,"c03237").rename("specific_humidity")
    sfc_q=sfc_q.sel(time=slice(t1,t2,3))
    # Factor to convert to hPa
    sfc_p=load_file(psfc_file,"c00409")/100.
    sfc_p=sfc_p.sel(time=slice(t1,t2,3)).rename("pressure").assign_attrs(units="hPa")
    toaolr=load_file(tolw_file,"a03332").rename("tolw")
    toaolr=toaolr.sel(time=slice(t1,t2,3))

    # Get dewpoint and equivalent potential temps
    sfc_td=mpcalc.dewpoint_from_specific_humidity(sfc_p.metpy.quantify(),sfc_t.metpy.quantify(),sfc_q.metpy.quantify()).metpy.convert_units('degK')
    sfc_td=sfc_td.rename("dew_point_temperature").assign_attrs(units="K").metpy.dequantify()
    sfc_te=mpcalc.equivalent_potential_temperature(sfc_p.metpy.quantify(),sfc_t.metpy.quantify(),sfc_td.metpy.quantify())
    sfc_te=sfc_te.rename("equivalent_potential_temperature").assign_attrs(units="K").metpy.dequantify()
    sfc=xr.merge([sfc_p,sfc_t,sfc_q,sfc_td,sfc_te]) # exclude OLR as not in prof
    # To merge surface and profile data use arbitrary vertical coordinate "level". Assign surface level=0
    sfc=sfc.assign_coords(level=0).drop_vars("height")

    # Thermodynamic profile fields
    tprf_file=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/f30204/f30204_A3hr_inst_?????_4km_{}0300-{}0000.nc".format(edate_str,ndate_str))[0]
    qprf_file=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/f30205/f30205_A3hr_inst_?????_4km_{}0300-{}0000.nc".format(edate_str,ndate_str))[0]
    prf_t=load_file(tprf_file,"f30204").rename("air_temperature").sel(time=slice(t1,t2)) # 3 hourly profile fields
    prf_q=load_file(qprf_file,"f30205").rename("specific_humidity").sel(time=slice(t1,t2))

    level_check=7
    if level != "all":
        prf_t=prf_t.sel(pressure=slice(level,level))
        prf_q=prf_q.sel(pressure=slice(level,level))
        level_check=2
    # Only t and q available from model output - calculate the rest
    # build up 2D fields of level pressure (i.e. constant; needed for metpy calculations)
    temp=np.ones(prf_t.shape)
    for idx, p in enumerate(prf_t.pressure.values):    
        temp[:,idx,:,:]=p*temp[:,idx,:,:]
    prf_p = xr.DataArray(temp,coords=prf_t.coords,dims=prf_t.dims).rename("p").assign_attrs(units="hPa")
    prf_td = mpcalc.dewpoint_from_specific_humidity(prf_p.values*units('hPa'),prf_t.values*units('K'),prf_q.values*units('g/kg')).to("degK")
    prf_te = mpcalc.equivalent_potential_temperature(prf_p.values*units('hPa'),prf_t.values*units('K'),prf_td)
    # Combine into single, multi-level dataset and prepare for concatenation with surface data
    prf=xr.merge([prf_p,prf_t,prf_q])
    prf["dew_point_temperature"]=(prf.dims,prf_td.magnitude)
    prf["equivalent_potential_temperature"]=(prf.dims,prf_te.magnitude)
    # Exchange vertical coordinates
    prf=prf.swap_dims({"pressure":"level"}).assign_coords(level=("level",np.arange(1,idx+2))).drop_vars("pressure")
    prf=prf.rename({"p":"pressure"})

    # Concatenate surface and profile fields on vertical axis
    # Surface files on different horizontal native UM grid to profiles!
    sfc=sfc.interp_like(prf)
    fullprofiles=xr.concat([sfc,prf],dim="level")
    # Apply surface pressure masking on lowest levels (up to 800hPa); not the most elegant way, oh well not a crux
    for p in range(1,level_check):
        for var in fullprofiles.data_vars:
            fullprofiles[var][:,:,:,p]=fullprofiles[var][:,:,:,p].where(fullprofiles.pressure[:,:,:,p] < fullprofiles.pressure[:,:,:,0])           
    fullprofiles["tbb"]=olr_to_bt(toaolr).interp_like(fullprofiles)

    return fullprofiles

#######################################################################################################
# Routine to identify convective cores within known MCS footprint from vertical velocity field 
# Inputs:
#     - row : row of MCS table dataframe
#     - bt : full-domain field of TOA brighntess temperature; must be on same grid as w500
#     - w500 : full-domain field of 500hPa upwards vertical velocity
#     - cores : bool, when False routine only returns initial w distribution information (fast)
# Note function designed for use with df.apply(func)
#######################################################################################################
def w500_vals(row,bt,w500,cores=True):
    #Code requires 2D arrays; select MCS time.
    bt=bt[bt.time.dt.hour==row.hour].isel(time=0)
    w500=w500[w500.time.dt.hour==row.hour].isel(time=0)
    # Zoom into MCS area
    fprint=bt.sel(latitude=slice(row["minlat"],row["maxlat"]),longitude=slice(row["minlon"],row["maxlon"]))
    # Isolate MCS features
    fprint=fprint.where(fprint<-50)
    # Isolate MCS vertical velocity field
    w500=w500.where(fprint)
    # Extract details on MCS w distribution
    row["max_w500"]=float(w500.max())
    row["p99.5_w500"]=float(w500.quantile(0.995))
    row["p99_w500"]=float(w500.quantile(0.99))
    row["p98_w500"]=float(w500.quantile(0.98))
    row["p95_w500"]=float(w500.quantile(0.95))
    row["p90_w500"]=float(w500.quantile(0.90))

    # Core identification
    def core_widths(w500,fprint,thld):
        #Get core objects - distinct regions with w500 > thld m/s.
        w_cores=w500.where(fprint).where(w500>thld).fillna(0)
        cores = label(w_cores.values)[0]
    
        #Exception in case finds no objects
        if len(np.unique(cores)) > 1:
            cores_filt=cores.copy()
            # Want to ultimately pick core with strongest updraft; filter out any pollution from single pixels
            for id in np.unique(cores):
                area=np.count_nonzero(cores_filt==id)
                if area<3:
                    cores_filt[cores_filt==id]=0
                
            if len(np.unique(cores_filt)) > 1:
                # Get label corresponding to core with strongest updraft; need where statement in case max w in a single-pixel updraft
                id_max=cores_filt.flatten()[np.argmax(np.where(cores_filt!=0,w_cores.values,0))]
                # Area is number of labeled pixels with id corresponding to wmax
                area=np.count_nonzero(cores_filt==id_max)
            # If no cores bigger than 2 pixels then just go with whichever is strongest
            else:
                # No where statement needed as w_cores all above thld
                id_max=cores.flatten()[np.argmax(w_cores.values)]
                area=np.count_nonzero(cores==id_max)
        else:
            area=np.NaN
        
        # Area currently in pixels - grid resolution for CP4 is 4.5km
        return 4.5**2 * area

    if cores==True:
        # Numerical threshold - not recommended for comparing across models, but useful physical gauge
        for thld in [5,10]:
            if float(w500.max())>thld:
                row["area_w500_%sms_core"%thld]=core_widths(w500,fprint,thld)
            else:
                row["area_w500_%sms_core"%thld]=np.NaN
        # Percentile based thresholds - recommended for comparing across models
        for pcle in [98,99,99.5]:
            row["area_w500_p%s_core"%pcle]=core_widths(w500,fprint,np.percentile(w500.values,pcle))

    return row

#######################################################################################################
# Routine to calculate core buoyancy from equivalent potential temperature fields
# Inputs:
#     - row : row of MCS table dataframe
#     - mcs_te : full-domain 500hPa equivalent potential temperature field at MCS time (K)
#     - env_te : full-domain saturated 500hPa " " " fields at environment time (K)
#     - w500 : full-domain field of 500hPa upwards vertical velocity (m/s)
#     - bt : full-domain field of TOA brighntess temperature (degC); must be on same grid as w500
# Note function designed for use with df.apply(func)
#######################################################################################################
def core_theta_measures(row,mcs_te,env_te,w500,bt):
    #Core identification requires 2D arrays; select MCS time.
    bt=bt[bt.time.dt.hour==row.hour].isel(time=0)
    w500=w500[w500.time.dt.hour==row.hour].isel(time=0)
    mcs_te=mcs_te[mcs_te.time.dt.hour==row["hour"]].isel(time=0)
    # See w500_vals routine for following lines' comments
    fprint=bt.sel(latitude=slice(row["minlat"]-0.1,row["maxlat"]+0.1),longitude=slice(row["minlon"]-0.1,row["maxlon"]+0.1))
    fprint=fprint.where(fprint<-50)
    w500=w500.where(fprint)

    thld=np.percentile(w500.values,99)
    w_cores=w500.where(w500>thld).fillna(0)
    cores=label(w_cores.values)[0]
    # Here need to get the coordinate location of the strongest updraft sampled in w500_vals, so as to sample env_te
    id_max=cores.flatten()[np.argmax(w_cores.values)]
    cores=np.where(cores==id_max,1,0)
    # Strongest updraft in w_cores will be same maximum as w500
    coords=np.where(w500.values==np.max(w500.values))
    coords=(w500.latitude.values[coords[0]],w500.longitude.values[coords[1]])
    #above can very occasionally pick out two nearby grid point! Thus take mean in case:
    coords=(np.mean(coords[0]),np.mean(coords[1]))
    
    # Isolate core equivalent potential temperature values
    core_te500=mcs_te.where(fprint).where(cores>0)
    # Sample corresponding environment - 0.7deg for consistency with "environment" definition
    theta_env=env_te.sel(latitude=slice(float(coords[0])-0.375,float(coords[0])+0.375),
                         longitude=slice(float(coords[1])-0.375,float(coords[1])+0.375)).mean()

    # buoyancy := difference between 500hPa core theta_e and saturated environmental value for core location.
    row["theta_e_core_buoy"] = float(core_te500.mean()) - float(theta_env)
    return row

#######################################################################################################
# Routine to calculate 0.7deg mean sounding centred on single MCS core location (tminlat/lon)
# Inputs:
#     - row : row of MCS table dataframe
#     - env : key parameter! env=0 = 18 or 21UTC; positive values sample env hours prior to !!18UTC!!. Must be multiples of 3.
# Note function designed for use with df.apply(func)
#######################################################################################################
def get_sounding(row,atm,env=0):
    # env2 positive integer (for precursor vals); env determines time when sounding sampled relative to 18UTC MCS time.
    env2=env+int(row["hour"]-18)
    # For 21UTC MCSs previous setup prevents sampling at storm time; for consistency between hours, redefine env=0 case so captures MCS time.
    if row["hour"]==21 and env==0:
        env2=0
    tstamp=atm.time[atm.time.dt.hour==(row["hour"]-env2)]
    
    # Restrict to location of MCS core, environmental resolution to capture meso \beta-scale profile.
    # Final time isel just to remove time coord and reduce to 2D grid; bit ugly, sorry
    prof=atm.sel(latitude=slice(row["tminlat"]-0.375,row["tminlat"]+0.375),longitude=slice(row["tminlon"]-0.375,row["tminlon"]+0.375),time=tstamp)
    # Must remove any NaNs before profile calculation!
    prof=prof.dropna(dim="level").mean(dim=["latitude","longitude"]).isel(time=0)
    # Metpy standard routines for parcel physics
    parc=mpcalc.parcel_profile(prof.pressure.values*units("hPa"),prof.air_temperature.isel(level=0).values*units("K"),prof.dew_point_temperature.isel(level=0).values*units("K"))
    lnb=mpcalc.el(prof.pressure.values*units("hPa"),prof.air_temperature.values*units("K"),prof.dew_point_temperature.values*units("K"),parc)
    # Exception in case parcel trajectory raises problems (rare)
    try:
        cpe=mpcalc.cape_cin(prof.pressure.values*units("hPa"),prof.air_temperature.values*units("K"),prof.dew_point_temperature.values*units("K"),parc)
        cpe=(cpe[0].magnitude,cpe[1].magnitude)
    except:
        cpe=(np.NaN,np.NaN)
    # Label MCS vs environmental profile values
    if env==0:
        m="_mcs"
        # Linear interpolation on parcel trajectory to get rough estimate of pressure-coordinate of anvils
        parc=xr.DataArray(prof.pressure.values,coords=[parc.magnitude],dims=["temperature"])
        row["p_mcs_anv"]=float(parc.interp(temperature=row["tmean"]+273.15).values)
    else:
        m=str(-env)
    row["plnb"+m]=lnb[0].magnitude
    row["tlnb"+m]=lnb[1].magnitude
    row["CAPE"+m]=cpe[0]
    row["CIN"+m]=cpe[1]
    return row

#######################################################################################################
# Available regions: "sahel", "wafrica", "safrica"
# Available years: 1997 - 2006 inclusive
# Important note: to facilitate parallelisation code only handles one year of CP4 data. Then run array job on batch server.
#######################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--region", required=False, type=str, default="wafrica")
parser.add_argument("-y", "--year", required=False, type=int, default=2006)
args = parser.parse_args()

# NOTE: ALL PATHS ARE AVAILABLE TO JASMIN impala gws USERS (contact: Elizabeth Kendon)

region=args.region.lower()
year=args.year
# To capture peak MCS activity, for S. Hemisphere want winter DYAMOND period (slightly tweaked to ensure 40 days from 360 day calendar)
if region=="safrica":
    period=xr.cftime_range(start="{}-01-20".format(year),end="{}-02-29".format(year),calendar="360_day")
    utc_offset=-3
else:
    period=xr.cftime_range(start="{}-08-01".format(year),end="{}-09-10".format(year),calendar="360_day")
    utc_offset=0
    
MCS_data=pd.read_csv("CP4_{}_{}_MCS_table.csv".format(year,region.capitalize()))

# Update only setup for Maybee et al relevant quantities.    
dict={"w_vals":[],"theta_e_core_buoy":[],"sndg_mcs":[],"sndg-3":[],"sndg-6":[]}

for date in period:
    print(date)
    # CP4 files names contain day and next day.
    edate_str="%04d%02d%02d" % (date.year,date.month,date.day)
    ndate = date + datetime.timedelta(days=1)
    ndate_str="%04d%02d%02d" % (ndate.year,ndate.month,ndate.day)

    t1,t2=time=slice(date.replace(hour=18+utc_offset),date.replace(hour=22+utc_offset)
    env_time=date.replace(hour=12+utc_offset)
    
    # Get relevant domain-wide BT and w500 fields - OLR is hourly but that's okay, handled in functions.
    # Load TOA OLR (W/m2)
    toaolr=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/a03332/a03332_A1hr_mean_?????_4km_{}0030-{}2330.nc".format(edate_str,edate_str))[0]
    toaolr=load_file(toaolr,"a03332").sel(time=slice(t1,t2))
    toaolr=olr_to_bt(toaolr).rename("bt")

    # For cores need 500hPa w (m/s) - CP4 native data only available for pressure velocity omega (Pa/s)
    om_file=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/f30208/f30208_A3hr_inst_?????_4km_{}0300-{}0000.nc".format(edate_str,ndate_str))[0]
    om_file=load_file(om_file,"f30208").rename("omega").sel(pressure=500,time=slice(t1,t2))
    # Conversion to w (metpy) requires T and P field
    # Load air temperature (K)
    tprf_file=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/f30204/f30204_A3hr_inst_?????_4km_{}0300-{}0000.nc".format(edate_str,ndate_str))[0]
    tprf_file=load_file(tprf_file,"f30204").rename("air_temperature").sel(pressure=500,time=slice(t1,t2))
    # Dummy pressure field
    pressure=500*np.ones(om_file.shape)

    # Convert omega to w.
    w_file = mpcalc.vertical_velocity(om_file.values * units("Pa/s"), pressure * units("hPa"), tprf_file * units("K"))
    w_file = xr.DataArray(w_file.magnitude,coords=om_file.coords)
    # OLR and w need to be on same grid; use OLR as that's ultimately the grid used for MCS identification
    w_file = w_file.interp(longitude=toaolr.longitude,latitude=toaolr.latitude)
    
    # Buoyancy measure requires environmental, fully saturated theta-e at 500hPa. Pick 12UTC for environment.
    # Load air temperature profiles (K)
    env_t=glob.glob("/gws/nopw/j04/impala/shared/CP4A/ncfiles/4km/f30204/f30204_A3hr_inst_?????_4km_{}0300-{}0000.nc".format(edate_str,ndate_str))[0]
    env_t=load_file(env_t,"f30204").rename("air_temperature").sel(time=env_time,pressure=500)
    # Saturated, so fix RH at 100%
    env_rh=100*np.ones(env_t.shape)
    p500=500*np.ones(env_t.shape)
    env_td=mpcalc.dewpoint_from_relative_humidity(env_t.values*units('K'),env_rh*units('percent')).to("degK")
    env_te=mpcalc.equivalent_potential_temperature(p500*units("hPa"),env_t.values*units("K"),env_td)
    # Environmental satured theta_e field - interpolate onto OLR grid.
    env_te=xr.DataArray(env_te.magnitude,env_t.coords).interp(longitude=toaolr.longitude,latitude=toaolr.latitude)
    
    # Now get storm-time actual theta-e values at 500hPa (level coord 12), via building full profile ready for soundings
    atm=build_native_profile(date)
    # MCS-times theta_e field:
    mcs_te=atm.sel(level=12).equivalent_potential_temperature.interp(longitude=toaolr.longitude,latitude=toaolr.latitude)

    # Restrict MCS table to relevant entries
    day_data=MCS_data[(MCS_data["day"]==date.day) & (MCS_data["month"]==date.month) & (MCS_data["hour"].isin([18+utc_offset,21+utc_offset]))]
    # Apply routines to get data on MCS cores
    dict["w_vals"].append(day_data.apply(w500_vals,args=(toaolr,w_file,),axis=1).filter(regex="w500"))
    dict["theta_e_core_buoy"].append(day_data.apply(core_theta_measures,args=(mcs_te,env_te,w_file,toaolr,),axis=1).filter(regex="theta_e_core"))

    # Save some memory before running soundings routines - those depend on atm only.
    del(w_file,toaolr,tprf_file,pressure,om_file)
    dict["sndg_mcs"].append(day_data[["tminlat","tminlon","hour","tmean"]].apply(get_sounding,args=(atm,),env=0,axis=1).filter(regex="_mcs"))
    dict["sndg-3"].append(day_data[["tminlat","tminlon","hour"]].apply(get_sounding,args=(atm,),env=3,axis=1).filter(regex="-3"))
    dict["sndg-6"].append(day_data[["tminlat","tminlon","hour"]].apply(get_sounding,args=(atm,),env=6,axis=1).filter(regex="-6"))

# Join everything together
for var in list(dict.keys()):
    dict[var]=pd.concat(dict[var])
    MCS_data=pd.concat([MCS_data,dict[var]],axis=1)

# Data always saved to same master file; should never overwrite entries. Be careful if updating.
MCS_data.to_csv("CP4_{}_{}_MCS_table.csv".format(year,region.capitalize()),index=False)
    
