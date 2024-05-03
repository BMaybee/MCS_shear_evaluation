import MCS_table_create
import numpy as np
import xarray as xr
import iris
import pandas as pd
import argparse
import time
import warnings
import metpy.calc as mpcalc
from metpy.units import units
from scipy.ndimage import label
import sys
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
    return tf - 273.15

# Routine for loading native grid pp files
def load_file(cube,region,pad=False):
    region=region.lower()
    if type(cube) == list:
        da=[]
        for c in cube:
            da.append(xr.DataArray.from_iris(c))
        ds=xr.concat(da,dim="time")
    else:
        ds=xr.DataArray.from_iris(cube)
    ds=ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds=ds.sortby(ds.longitude)
    try:
        # Looking over land mostly at upper levels - remove 1000hPa pressure level.
        ds=ds.sel(pressure=slice(None,925))
        ds=ds.isel(pressure=slice(None,None,-1))
    except:
        pass

    if pad!=False:
        pad=0.1
    else:
        pad=0
    
    if region == "full":
        pass
    # Sahel region domain from Senior et al, 2021
    if region == "sahel":
        ds=ds.sel(latitude=slice(9,19),longitude=slice(-12,12))
    # Following regions align with Barton et al global MCS + soil moisture analysis
    if region == "wafrica":
        ds=ds.sel(latitude=slice(4,25),longitude=slice(-18,25))
    if region == "safrica":
        ds=ds.sel(latitude=slice(-35,-15),longitude=slice(20,35))
    if region == "samerica":
        ds=ds.sel(latitude=slice(-40,-20.5),longitude=slice(-68,-47))
    if region == "india":
        ds=ds.sel(latitude=slice(5,30),longitude=slice(70,90))
    if region == "aus":
        ds=ds.sel(latitude=slice(-23,-11),longitude=slice(120,140))
        
    return ds


#######################################################################################################
# Routine to build full depth atmospheric thermoydnamic profiles
# Profile constructed for calendar day between environmental time and latest MCS-time profile field
# Profiles built from pressure level data but include surface fields
#######################################################################################################
def build_native_profile(date):
    # Need to find right native field. Files cover 12 hour periods, starting from hour after file name
    # Thus offset hour by 1 before getting relevant 12 hour timestep.
    # For Wafrica, this means environmental and MCS-relevant fields are in different files!
    env_tstep=int((date-period[0]).total_seconds()//3600 / 12) * 12
    mcs_tstep=env_tstep+12
    t1=date+pd.Timedelta(12+utc_offset,"h")
    t2=date+pd.Timedelta(21+utc_offset,"h")

    sfc_file1=iris.load(ffile+"a%03d.pp"%(env_tstep))
    sfc_file2=iris.load(ffile+"a%03d.pp"%(mcs_tstep))

    # Variables to include in profile field:
    #     - t : air temperature (K)
    #     - td : dewpoint temperature (K)
    #     - rh : relative humidity (%)
    #     - te : equivalent potential temperature (K)
    #     - p : air pressure (hPa)
    sfc_t=load_file([sfc_file1[0],sfc_file2[0]],region).sel(time=slice(t1,t2,3)) # 3 hourly profile fields but hourly sfc data
    sfc_td=load_file([sfc_file1[1],sfc_file2[1]],region).sel(time=slice(t1,t2,3)) # " " "
    sfc_rh=load_file([sfc_file1[2],sfc_file2[2]],region).sel(time=slice(t1,t2,3)) # " " "
    sfc_p=load_file([sfc_file1[3],sfc_file2[3]],region).sel(time=slice(t1,t2))/100. # sfcp is outputted on 3 hourly stream; factor converts to hPa
    sfc_p=sfc_p.rename("pressure").assign_attrs(units="hPa")
    # Save some memory...
    del sfc_file1, sfc_file2

    sfc_te=mpcalc.equivalent_potential_temperature(sfc_p.metpy.quantify(),sfc_t.metpy.quantify(),sfc_td.metpy.quantify())
    sfc_te=sfc_te.rename("equivalent_potential_temperature").assign_attrs(units="K").metpy.dequantify()
    sfc=xr.merge([sfc_p,sfc_t,sfc_rh,sfc_td,sfc_te])
    # To merge surface and profile data use arbitrary vertical coordinate "level". Assign surface level=0
    sfc=sfc.assign_coords(level=0).drop_vars("height")

    # Thermodynamic profile fields in native d stream
    tprf_file1=iris.load(ffile+"d%03d.pp"%(env_tstep))
    tprf_file2=iris.load(ffile+"d%03d.pp"%(mcs_tstep))
    
    # Only t and rh available from model output - calculate the rest
    prf_t=load_file([tprf_file1[0],tprf_file2[0]],region).sel(time=slice(t1,t2)) # 3 hourly profile fields
    prf_rh=load_file([tprf_file1[2],tprf_file2[2]],region).sel(time=slice(t1,t2)) # " "
    # Build up 2D fields of level pressure (i.e. constant; needed for metpy calculations)
    temp=np.ones(prf_t.shape)
    for idx, p in enumerate(prf_t.pressure.values):    
        temp[:,idx,:,:]=p*temp[:,idx,:,:]
    prf_p=xr.DataArray(temp,coords=prf_t.coords,dims=prf_t.dims).rename("p").assign_attrs(units="hPa")
    
    prf_td = mpcalc.dewpoint_from_relative_humidity(prf_t.values*units('K'),prf_rh.values*units('percent')).to("degK")
    prf_te = mpcalc.equivalent_potential_temperature(prf_p.values*units('hPa'),prf_t.values*units('K'),prf_td)
    #Combine into single, multi-level dataset and prepare for concatenation with surface data
    prf=xr.merge([prf_p,prf_t,prf_rh])
    prf["dew_point_temperature"]=(prf.dims,prf_td.magnitude)
    prf["equivalent_potential_temperature"]=(prf.dims,prf_te.magnitude)
    # Exchange vertical coordinates
    prf=prf.swap_dims({"pressure":"level"}).assign_coords(level=("level",np.arange(1,len(prf_t.pressure)+1))).drop_vars("pressure")
    prf=prf.rename({"p":"pressure"})

    # Concatenate surface and profile fields on vertical axis
    fullprofiles=xr.concat([sfc,prf],dim="level")
    # Apply surface pressure masking on lowest levels (up to 800hPa); not the most elegant way, oh well not a crux
    for p in [1,2,3,4]:
        for var in fullprofiles.data_vars:
            fullprofiles[var][:,:,:,p]=fullprofiles[var][:,:,:,p].where(fullprofiles.pressure[:,:,:,p] <= fullprofiles.pressure[:,:,:,0])
    # All done - but IMPORTANT NOTE: previous step means these profiles are NOT APPROPRIATE FOR VERTICAL INTEGRATION
    # Existence of NaN values, stemming from .where, will pollute integrals.
    # If remove masking and integrating, note important to account for coordinate change of variables.
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
    # Zoom into MCS area; pad to account for initial identification on 0.1 deg grid
    fprint=bt.sel(latitude=slice(row["minlat"]-0.1,row["maxlat"]+0.1),longitude=slice(row["minlon"]-0.1,row["maxlon"]+0.1))
    # Isolate MCS features
    fprint=fprint.where(fprint<-50)
    # Isolate MCS vertical velocity field
    w500=w500.where(fprint)
    # Extract details on MCS w distribution
    row["max_w500"]=float(w500.max())
    row["p99_w500"]=float(w500.quantile(0.99))
    row["p95_w500"]=float(w500.quantile(0.95))
    row["p90_w500"]=float(w500.quantile(0.90))
    row["p98_w500"]=float(w500.quantile(0.98))
    row["p99.5_w500"]=float(w500.quantile(0.995))

    # Core identification
    def core_widths(w500,fprint,thld):
        #Get core objects - distinct regions with w500 > thld m/s.
        w_cores=w500.where(fprint).where(w500>thld).fillna(0)
        cores = label(w_cores.values)[0]
    
        #Exception in case finds no objects - quite common with numerical threshold (eg in parameterised model)
        if len(np.unique(cores)) > 1:
            cores_filt=cores.copy()
            # Want to ultimately pick core with strongest updraft; filter out any pollution from single pixels
            for id in np.unique(cores):
                area=np.count_nonzero(cores_filt==id)
                if area<3:
                    cores_filt[cores_filt==id]=0
                
            if len(np.unique(cores_filt)) > 1:
                # Get label corresponding to core with strongest updraft; need where statement in case max w is in a single-pixel updraft
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

        # Area currently in pixels - convert to km2 using global res_factor
        return res_factor * area

    if cores==True:
        # Numerical thresholds - not recommended for comparing across models, but useful physical gauge
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
    # Core identification requires 2D arrays; select MCS time.
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
    prof=atm.sel(latitude=slice(row["tminlat"]-0.375,row["tminlat"]+0.375),longitude=slice(row["tminlon"]-0.375,row["tminlon"]+0.375),time=tstamp).isel(time=0)
    prof=prof.mean(dim=["latitude","longitude"])
    # Must remove any NaNs before profile calculation!
    prof=prof.where(prof.pressure<=prof.pressure[0]).dropna(dim="level")
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
# Available regions: "sahel", "wafrica", "safrica", "samerica", "india", "aus"
# Available simulations: "channel_RAL3p2","channel_GAL9","lam" (n.b. only n2560 and 2.2km grids presently)
# Available update routines:
#     - soundings : runs get_sounding routine for 12Z + utc_offset, 15Z +  utc_offset and MCS-times
#     - cores : runs w500_vals and core_theta_e_measures routines for MCS-times
#######################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--region", required=True, type=str)
parser.add_argument("-s", "--simulation", required=False, default="channel_RAL3p2", type=str)
parser.add_argument("-u", "--update", required=True, default=type=str)
parser.add_argument("-sc", "--source", required=False, default="nat")
args = parser.parse_args()

region=args.region
sim=args.simulation
source=args.source
# To capture peak MCS activity, for N. Hemisphere regions want summer DYAMOND period
if region == "sahel" or region == "wafrica" or region == "india":
    period=pd.date_range(start="2016-08-01",end="2016-09-09")
# And vice versa for SH...
if region == "samerica" or region == "safrica" or region == "aus":
    period=pd.date_range(start="2020-01-20",end="2020-02-28")

if region == "sahel" or region =="wafrica":
    utc_offset = 0   
if region == "safrica":
    utc_offset = -3
if region == "samerica":
    utc_offset = +3
if region == "india":
    utc_offset = -6
if region == "aus":
    utc_offset = -9

#######################################################################################################
# DATA SPECIFIC SETTINGS - FFILE PATHS AVAILABLE TO ALL JASMIN kscale gws USERS (contact: Huw Lewis)

# Res_factors for area conversions; treated as constant; variation of longitude length with latitude very minor for Wafrica
if sim == "channel_RAL3p2": # Trop5-exp
    ffile="/gws/nopw/j04/kscale/bmaybee/CTC_n2560_RAL3p2/20160801T0000Z_CTC_n2560_RAL3p2_pver"
    res_name,res_factor="n2560",7.6*5.2 #7.6 km is specific to Wafrica! ~12N length of longitudinal grid spacing.
elif sim == "channel_GAL9": # Trop5-param
    ffile="/gws/nopw/j04/kscale/bmaybee/CTC_n2560_GAL9/20160801T0000Z_CTC_n2560_GAL9_pver"
    res_name,res_factor=5,"n2560",7.5*5.2 # " "
elif sim == "lam_RAL3p2": # LAM2.2, NH Africa only
    ffile="/gws/nopw/j04/kscale/bmaybee/Africa_km2p2_RAL3p2/20160801T0000Z_Africa_km2p2_RAL3p2_pver"
    res_name,res_factor="km2p2",2.2**2 
#######################################################################################################
    
#Load in MCS table to be updated
MCS_data=pd.read_csv('{}_{}_{}_{}_MCS_table_{}.csv'.format(sim.split("_")[0].capitalize(),res_name,sim.split("_")[-1],region.capitalize()))

if update=="soundings":
    dict={}
    col=[]
    for var in ["sndg-3","sndg-6","sndg_mcs"]:
        dict[var]=[]
    
    for i, date in enumerate(period):
        print(date)
        # Get thermodynamic atmospheric profile data - covers between 12Z+utc_offset and 21Z+utc_offset
        atm=build_native_profile(date)

        # Restrict to MCS times on day
        day_data=MCS_data[(MCS_data["day"]==date.day) & (MCS_data["month"]==date.month) & (MCS_data["hour"].isin([18+utc_offset,21+utc_offset]))]
        # Apply get_sounding routine to day's data
        if len(day_data)>0:
            dict["sndg_mcs"].append(day_data[["tminlat","tminlon","hour","tmean"]].apply(get_sounding,env=0,axis=1).filter(regex="_mcs"))
            dict["sndg-3"].append(day_data[["tminlat","tminlon","hour"]].apply(get_sounding,env=3,axis=1).filter(regex="-3"))
            dict["sndg-6"].append(day_data[["tminlat","tminlon","hour"]].apply(get_sounding,env=6,axis=1).filter(regex="-6"))

    # Merge output values into MCS tables
    for var in list(dict.keys()):
        dict[var]=pd.concat(dict[var])
        MCS_data=pd.concat([MCS_data,dict[var]],axis=1)


if update=="cores":
    dict={"w_vals":[],"theta_vals":[]}

    for date in period:
        # Need to find right native field. Files cover 12 hour periods, starting from hour after file name
        # Thus offset hour by 1 before getting relevant 12 hour timestep.
        # For Wafrica, this means environmental and MCS-relevant fields are in different files!
        edate=date+pd.Timedelta(12+utc_offset,"h")
        # Relevant MCS times (3 hourly profile fields) - note use 10, not 9, to include 21UTC in slices.
        t1,t2=edate+pd.Timedelta(6,"h"),edate+pd.Timedelta(10,"h")
        mcs_tstep=int((edate+pd.Timedelta(6,"h")-period[0]).total_seconds()//3600 / 12) * 12
        env_tstep=int((edate-pd.Timedelta(1,"h")-period[0]).total_seconds()//3600 / 12) * 12

        # Get relevant domain-wide BT and w500 fields - OLR is hourly but that's okay, handled in functions.
        # All variables loaded are on same native UM grids (grid_t)
        # Load TOA OLR (W/m2)
        toaolr=iris.load(ffile+"a%03d.pp"%(mcs_tstep))
        toaolr=load_file(toaolr[4],region).sel(time=slice(t1,t2))
        toaolr=olr_to_bt(toaolr)
        # Load w profiles (m/s)
        w_file=iris.load(ffile+"c%03d.pp"%(mcs_tstep))
        w_file=load_file(w_file[0],region).sel(pressure=500,time=slice(t1,t2))

        # Buoyancy measure requires domain-wide 500hPa equivalent potential temperature fields
        # Firstly environmental fully saturated theta-e
        env_prf=iris.load(ffile+"d%03d.pp"%(env_tstep))
        # Load air temperature profiles (K)
        env_t=load_file(env_prf[0],region).sel(pressure=500,time=slice(t1,t2))
        # Saturated, so fix RH at 100%
        env_rh=100*np.ones(env_t.shape)
        # Dummy pressure field for metpy routines
        p500=500*np.ones(env_t.shape)
        env_td=mpcalc.dewpoint_from_relative_humidity(env_t.values*units('K'),env_rh*units('percent')).to("degK")
        env_te=mpcalc.equivalent_potential_temperature(p500*units("hPa"),env_t.values*units("K"),env_td)
        # Environmental satured theta_e field:
        env_te=xr.DataArray(env_te.magnitude,env_t.coords)

        # Now MCS time theta_e fields, with actual humidity values
        mcs_prf=iris.load(ffile+"d%03d.pp"%(mcs_tstep))
        # Load air temperature profiles (K)
        mcs_t=load_file(mcs_prf[0],region).sel(time=slice(t1,t2),pressure=500)
        # Load relative humidity profiles (%)
        mcs_rh=load_file(mcs_prf[2],region).sel(time=slice(t1,t2),pressure=500)
        mcs_td=mpcalc.dewpoint_from_relative_humidity(mcs_t.values*units('K'),mcs_rh.values*units('percent')).to("degK")
        mcs_te=mpcalc.equivalent_potential_temperature(p500*units("hPa"),mcs_t.values*units("K"),mcs_td)
        # MCS-times theta_e field:
        mcs_te=xr.DataArray(mcs_te.magnitude,mcs_t.coords)

        # Restrict MCS table to relevant entries
        day_data=MCS_data[(MCS_data["day"]==date.day) & (MCS_data["month"]==date.month) & (MCS_data["hour"].isin([18+utc_offset,21+utc_offset])]
        # Apply routines to get data on MCS cores
        dict["w_vals"].append(day_data.apply(w500_vals,args=(toaolr,w_file,),axis=1).filter(regex="w500"))
        dict["theta_vals"].append(day_data.apply(core_theta_measures,args=(mcs_te,env_te,w_file,toaolr,),axis=1).filter(regex="theta_e_core"))

    # Join everything together
    for var in list(dict.keys()):
        dict[var]=pd.concat(dict[var])
        MCS_data=pd.concat([MCS_data,dict[var]],axis=1)

# Data always saved to same master file; should never overwrite entries. Be careful if updating.
MCS_data.to_csv('{}_{}_{}_{}_MCS_table_{}.csv'.format(sim.split("_")[0].capitalize(),res_name,sim.split("_")[-1],region.capitalize()),index=False)