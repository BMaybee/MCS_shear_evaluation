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

# Routine for loading common JASMIN analysis grid fields
def load_file(ffile,var):
    if var=="all":
        ds=xr.open_dataset(ffile)
    else:
        ds=xr.open_dataset(ffile)[var]
        
    try:
        test=ds.latitude.min()
    except:
        ds=ds.rename({"lat":"latitude","lon":"longitude"})

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

# Routine for loading native grid pp files
def load_nfile(cube,region,pad=False):
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
        # Looking over land - remove 1000hPa pressure level.
        ds=ds.sel(pressure=slice(None,950))
        ds=ds.isel(pressure=slice(None,None,-1))
    except:
        pass

    if pad!=False:
        pad=0.1
    else:
        pad=0
    
    if region == "full":
        pass
    if region == "sahel":
        ds=ds.sel(latitude=slice(9,19+pad),longitude=slice(-12,12+pad))
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
# Primary function for identifying MCS snapshots and building table. Res is approximate model resolution, used to determine pixel threshold cf 5000km2
# Where possible use 0.1 deg ~ 11km grid spacing to align with Meteosat BT observations.
# Function covers one timestep only; table rows in basic_tab/tab objects.
#######################################################################################################
def make_table(date,hour,res=11):
    hour = hour + utc_offset
    if hour > 23:
        edate_str="%04d%02d%02d" % (date.year,date.month,date.day)
        sdate = date + pd.Timedelta(1,"d")
        sdate_str="%04d%02d%02d" % (sdate.year,sdate.month,sdate.day)
        hour = hour % 24
    else:
        edate_str="%04d%02d%02d" % (date.year,date.month,date.day)
        sdate_str=edate_str

    # Extract OLR fields (W/m2) and convert to BT (degC)
    olr_file=root+'single_olwr/{}_{}_{}_olwr_hourly.nc'.format(sdate_str,dymnd_run,domain)
    # Note use of isel - files span 0-23UTC
    olr=load_file(olr_file,"toa_outgoing_longwave_flux").isel(time=hour)
    olr = olr_to_bt(olr)
    # Mask out ocean areas - only want continental MCSs. NC4 file loads weirdly, hence roundabout masking!
    # Extra latitude slice needed to account for cases where domain box goes beyond simulation domain
    landseamask = load_file('/gws/nopw/j04/kscale/DATA/GPM_IMERG_LandSeaMask.2.nc4','landseamask').T.sel(latitude=slice(olr.latitude.min()-0.05,olr.latitude.max()+0.05))
    mask = olr.copy()
    mask.values = landseamask.values
    olr = olr.where(mask!=100)
    # Identification of MCSs from BT field. Constraints are Tb <= -50C, area >= 5000/res**2
    basic_tab = MCS_table_create.process_tir_image(olr, res)

    # Sample rainfall records (mm/hr) for MCSs at detection time
    precip_file=root+'precip/{}_{}_{}_precip_hourly.nc'.format(sdate_str,dymnd_run,domain)
    precip=load_file(precip_file,"precipitation_rate")
    tab = MCS_table_create.add_environment_toTable(precip,basic_tab,date,envvar_take=[],rainvar=True)

    """
    ### Population of 12UTC MCS environments from JASMIN gws fields, sampled at 0.7deg. Only 0.5deg for profile fields, not recommended ###
    pfile=root+"single_surp/{}_{}_{}_surp_hourly.nc".format(edate_str,dymnd_run,domain)
    env=load_file(pfile,"surface_air_pressure")/100.
    #BL environment: load variables for 12LT same day. q, twcw zonal shear, surface p and T.
    profile_650=root+'profile_650/{}_{}_{}_profile_3hourly_650_05deg.nc'.format(edate_str,dymnd_run,domain)
    profile_850=root+'profile_850/{}_{}_{}_profile_3hourly_850_05deg.nc'.format(edate_str,dymnd_run,domain)
    profile_925=root+'profile_925/{}_{}_{}_profile_3hourly_925_05deg.nc'.format(edate_str,dymnd_run,domain)
    tcw=root+'single_qtot/{}_{}_{}_qtot_3hourly.nc'.format(edate_str,dymnd_run,domain)
    u_sfc=root+'single_uwnd/{}_{}_{}_uwnd_hourly.nc'.format(edate_str,dymnd_run,domain)
    v_sfc=root+'single_vwnd/{}_{}_{}_vwnd_hourly.nc'.format(edate_str,dymnd_run,domain)
    #tcw_file=qtot is equivalent (Huw email), units appear to be mm
    env_u650=load_file(profile_650,"x_wind")
    sfcp=env.interp_like(env_u650)
    env_u850=load_file(profile_850,"x_wind").where(sfcp>850)
    env_u925=load_file(profile_925,"x_wind").where(sfcp>925)
    #env_usfc=load_file(u_sfc,"x_wind")
    #env_v650=load_file(profile_650,"y_wind")
    #env_v925=load_file(profile_925,"y_wind")
    #env_vsfc=load_file(v_sfc,"y_wind")
    try:
        env_q850=load_file(profile_850,"specific_humidity").where(sfcp>850)
        env_q925=load_file(profile_925,"specific_humidity").where(sfcp>925)
    except:
        env_q850=load_file(profile_850,"specific_humidity_calc").where(sfcp>850)
        #env_q925=load_file(profile_925,"specific_humidity_calc").where(sfcp>925)
    env_tcw=load_file(tcw,"total_column_q")
    #print(hour,12+utc_offset,env_q)
    #tab = MCS_table_create.add_environment_toTable(env_q925,tab,date,envvar_take=['q925'],env_hour=12+utc_offset)
    tab = MCS_table_create.add_environment_toTable(env_q850,tab,date,envvar_take=['q850'],env_hour=12+utc_offset)
    tab = MCS_table_create.add_environment_toTable(env_tcw,tab,date,envvar_take=['tcw'],env_hour=12+utc_offset)
    #tab = MCS_table_create.add_environment_toTable(env_u650,tab,date,envvar_take=['u650'],env_hour=12+utc_offset)
    #tab = MCS_table_create.add_environment_toTable(env_u925,tab,date,envvar_take=['u925'],env_hour=12+utc_offset)
    #tab = MCS_table_create.add_environment_toTable(env_usfc,tab,date,envvar_take=['usfc'],env_hour=12+utc_offset)
    #tab = MCS_table_create.add_environment_toTable(env_v650,tab,date,envvar_take=['v650'],env_hour=12+utc_offset)
    #tab = MCS_table_create.add_environment_toTable(env_v925,tab,date,envvar_take=['v925'],env_hour=12+utc_offset)
    #tab = MCS_table_create.add_environment_toTable(env_vsfc,tab,date,envvar_take=['vsfc'],env_hour=12+utc_offset)
    tab = MCS_table_create.add_environment_toTable(env_u650-env_u925,tab,date,envvar_take=['ushear650_925'],env_hour=12+utc_offset)
    tab = MCS_table_create.add_environment_toTable(env_u650-env_u850,tab,date,envvar_take=['ushear650_850'],env_hour=12+utc_offset)
    
    tfile=root+"single_tsfc/{}_{}_{}_tsfc_hourly.nc".format(edate_str,dymnd_run,domain)
    ufile=root+"single_uwnd/{}_{}_{}_uwnd_hourly.nc".format(edate_str,dymnd_run,domain)
    var="psfc"
    tab=MCS_table_create.add_environment_toTable(env,tab,date,envvar_take=[var],env_hour=12+utc_offset)
    var="surface_t"
    env=load_file(tfile,"surface_temperature")
    tab=MCS_table_create.add_environment_toTable(env,tab,date,envvar_take=[var],env_hour=12+utc_offset)
    var="ushear650_10m"
    u650=load_file(profile_650,"x_wind").isel(time=4+(utc_offset%3))
    usfc=load_file(ufile,"x_wind").isel(time=11+utc_offset)
    usfc=usfc.interp(latitude=u650.latitude,longitude=u650.longitude)
    env=(u650-usfc).assign_coords(time=env.time[11+utc_offset]).expand_dims("time")
    tab=MCS_table_create.add_environment_toTable(env,tab,date,envvar_take=[var],env_hour=12+utc_offset)
    """

    ### Population of 12UTC MCS environments from native grid fields (multiple grids), sampled at 0.7deg.###
    edate = date.replace(hour=12+utc_offset)
    # Need to find right native field. Files cover 12 hour periods, starting from hour after file name
    # Thus offset hour by 1 before getting relevant 12 hour timestep.
    env_tstep=int((edate-pd.Timedelta(1,"h")-period[0]).total_seconds()//3600 / 12) * 12

    # Get surface pressure field to mask out sub-surface values (interpolated in model output, must be removed)
    surp = load_nfile(iris.load(ffile+"a%03d.pp"%(env_tstep))[3],region) 
    surp = surp[surp.time.dt.hour==12+utc_offset]/100. # factor to convert Pa -> hPa
    # Environment: load variables for 12LT same day
    env_profsd = iris.load(ffile+"c%03d.pp"%(env_tstep)) # c stream, profiles, mostly dynamics: upward air velocity, wet bulb theta, x wind and y wind
    env_profst = iris.load(ffile+"d%03d.pp"%(env_tstep)) # d stream, profiles, mostly thermodynamics: air T, geopotential height, relative humidity

    # Thermodynamics first - on same grid as surp
    env_t=load_nfile(env_profst[0],region).sel(pressure=[925,850])
    env_rh=load_nfile(env_profst[2],region).sel(pressure=[925,850])
    # Get specific humidity values. Not in model diagnostics - calculate.
    # Restrict to 12UTC (for W Africa)
    ehr=12+utc_offset
    env_t = env_t[env_t.time.dt.hour==ehr]
    env_rh = env_rh[env_rh.time.dt.hour==ehr]
    env_td=mpcalc.dewpoint_from_relative_humidity(env_t*units("K"), env_rh*units("percent"))
    # Build dummy constant pressure fields for use by metpy routine
    pfield=np.ones(env_t.shape)
    pfield[0,0,:,:]=925*pfield[0,0,:,:]
    pfield[0,1,:,:]=850*pfield[0,1,:,:]
    env_q=mpcalc.specific_humidity_from_dewpoint(pfield*units("hPa"),env_td)
    env_q850=env_q.metpy.dequantify().sel(pressure=850).where(surp>850)
    env_q925=env_q.metpy.dequantify().sel(pressure=925).where(surp>925)
    # Get total column water; diagnostic outputted over model levels, no need to mask.
    env_tcw=load_nfile(iris.load(ffile+"b%03d.pp"%(env_tstep))[0],region)

    # Dynamics fields on different grid, requires regridding of surp. Zonal winds only.
    env_u=load_nfile(env_profsd[2],region)
    env_u650=env_u.sel(pressure=650)
    surp=surp.interp(latitude=env_u650.latitude,longitude=env_u650.longitude)
    env_u850=env_u.sel(pressure=850).where(surp>850)
    env_u925=env_u.sel(pressure=925).where(surp>925)
    
    # Routines to sample environmental fields for MCSs detected in table
    tab = MCS_table_create.add_environment_toTable(env_q925,tab,date,envvar_take=['q925'],env_hour=ehr)
    tab = MCS_table_create.add_environment_toTable(env_q850,tab,date,envvar_take=['q850'],env_hour=ehr)
    tab = MCS_table_create.add_environment_toTable(env_tcw,tab,date,envvar_take=['tcw'],env_hour=ehr)
    tab = MCS_table_create.add_environment_toTable(env_u650-env_u925,tab,date,envvar_take=['ushear650_925'],env_hour=ehr)
    tab = MCS_table_create.add_environment_toTable(env_u650-env_u850,tab,date,envvar_take=['ushear650_850'],env_hour=ehr)
    tab = MCS_table_create.add_environment_toTable(surp,tab,date,envvar_take=['psfc'],env_hour=ehr)

    # Only keep tabulated data.
    tab.pop('cloudMask')
    tab.pop('tir')
    return pd.DataFrame(tab)


#######################################################################################################
# Routine for parameterised models only; repeats initial steps of primary routine but uses convective-scheme rainfall only in building tab
# n.b. convective rainfall = STASH ID m01s05i205
# Merge output with primary table on storm_id
#######################################################################################################
def get_scheme_rainfall(domain,ffile,res=11):
    mcs_records=[]
    for date in period:
        print(date)
        for hour in np.arange(16,23):    
            hour = hour + utc_offset
            if hour > 23:
                edate_str="%04d%02d%02d" % (date.year,date.month,date.day)
                sdate = date + pd.Timedelta(1,"d")
                sdate_str="%04d%02d%02d" % (sdate.year,sdate.month,sdate.day)
                hour = hour % 24
            else:
                edate_str="%04d%02d%02d" % (date.year,date.month,date.day)
                sdate_str=edate_str
            
            olr_file=root+'single_olwr/{}_{}_{}_olwr_hourly.nc'.format(sdate_str,dymnd_run,domain)
            olr=load_file(olr_file,"toa_outgoing_longwave_flux").isel(time=hour)
            # convert olr to bt
            olr = olr_to_bt(olr) #
            # Mask out ocean areas - only want continental MCSs. NC4 file loads weirdly, hence roundabout masking!
            # Extra latitude slice needed to account for cases where domain box goes beyond simulation domain
            landseamask = load_file('/gws/nopw/j04/kscale/DATA/GPM_IMERG_LandSeaMask.2.nc4','landseamask').T.sel(latitude=slice(olr.latitude.min()-0.05,olr.latitude.max()+0.05))
            mask = olr.copy()
            mask.values = landseamask.values
            olr = olr.where(mask!=100)
            #Second parameter = approximate data resolution, for applying MCS areas.
            basic_tab = MCS_table_create.process_tir_image(olr, res)

            # see line 185
            tstep=int((date-period[0]).total_seconds()//3600 / 12) * 12 + 12
            # Native rainfall in b stream output
            precip_file=ffile+'b_convrain%03d.pp'%tstep
            precip=load_nfile(iris.load(precip_file)[1],region)
            #native grid precip is in kg/m2
            precip=3600*precip.interp(latitude=olr.latitude.values,longitude=olr.longitude.values)
            tab = MCS_table_create.add_environment_toTable(precip,basic_tab,date,envvar_take=[],rainvar=True)

            tab.pop('cloudMask')
            tab.pop('tir')
            mcs_records.append(pd.DataFrame(tab))
    
    mcs_records=pd.concat(mcs_records)
    domain=args.domain.capitalize()
    mcs_records.to_csv('{}_{}_GAL9_{}_MCS_convrain.csv'.format(domain,res_name,region.capitalize()),index=False)


#######################################################################################################
# Available regions: "sahel", "wafrica", "safrica", "samerica", "india", "aus"
# Available simulations: "channel", "lam", "global"
# Available resolutions: "n1280", "n2560", "km4p4", "km2p2"
# Available configurations: "RAL3p2" [CONVECTION PERMITTING] ; "GAL9" [PARAMETERISED CONVECTION]
# Available driving model physics configurations: GAL9 ; RAL3p2 [not considered - for Met Office evaluation work]
#######################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--region", required=True, type=str)
parser.add_argument("-d", "--domain", required=False, default="channel", type=str)
parser.add_argument("-res", "--res", required=False, default="n2560", type=str)
parser.add_argument("-c", "--config", required=False, default="RAL3p2", type=str)
parser.add_argument("-dm", "--driving_model", required=False, type=str, default="GAL9")
args = parser.parse_args()

region=args.region.lower()
domain=args.domain.lower()
res_name=args.res
config=args.config
dm=args.driving_model

# To capture peak MCS activity, for N. Hemisphere regions want summer DYAMOND period
if region == "sahel" or region == "wafrica" or "india" in region:
    period=pd.date_range(start="2016-08-01",end="2016-09-09")
    dymnd_run="20160801T0000Z"
# And vice versa for SH...
if region == "samerica" or region == "safrica" or region == "aus":
    period=pd.date_range(start="2020-01-20",end="2020-02-28")
    dymnd_run="20200120T0000Z"

# Approximate offsets from central meridian - minimum resolution 3h due to profile field output frequency
if region == "sahel" or region =="wafrica":
    utc_offset = 0   
if region == "safrica":
    utc_offset = -3
if region == "samerica":
    utc_offset = +3
if "india" in region:
    utc_offset = -6
if region == "aus":
    utc_offset = -9

#######################################################################################################
# DATA SPECIFIC SETTINGS - ALL PATHS AVAILABLE TO JASMIN kscale gws USERS (contact: Huw Lewis)

if domain!="lam":
    # Path for all analysis grid (common, 0.1deg) CTC data
    root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280{}/{}_{}_{}/'.format(dymnd_run,dm,domain,res_name,config)
    # Path for selected relevant native grid (n1280 or n2560) CTC data
    # AT N2560, THESE ARE TROP5 IN MAYBEE ET AL
    ffile="/gws/nopw/j04/kscale/bmaybee/CTC_{}_{}/20160801T0000Z_CTC_{}_{}_pver".format(config,res_name,res_name,config)
else:
    # Multiple LAMs available in hierarchy
    # Africa LAM [LAM2.2 IN MAYBEE ET AL]
    if region == "wafrica" or region == "safrica" or region == "sahel":
        # Analysis grid path
        root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280{}/lam_africa_{}_{}/'.format(dymnd_run,dm,res_name,config)
        # Native grid path
        ffile="/gws/nopw/j04/kscale/bmaybee/Africa_km2p2_RAL3p2/20160801T0000Z_Africa_km2p2_RAL3p2_pver"
        domain="africa"
    # South East Asia LAM - covers MCS-relevant region of Australia
    elif region == "aus":
        # Analysis grid path only - native grid data not pulled from MASS
        root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280{}/lam_sea_{}_{}/'.format(dymnd_run,dm,res_name,config)
        domain="sea"
    # South America and India LAMs
    else:
        # Analysis grid path only - native grid data not pulled from MASS
        root='/gws/nopw/j04/kscale/DATA/outdir_{}/DMn1280{}/lam_{}_{}_{}/'.format(dymnd_run,dm,region,res_name,config)
        domain=region
#######################################################################################################

if args.update is None:
    print(region,domain,res_name,config)
    # Populate list with hourly dataframes
    mcs_records=[]
    # Cycle through relevant DYAMOND period
    for date in period:
        print(date)
        # Sample MCSs hourly between 16 and 22UTC (inclusive)
        for hour in np.arange(16,23):
            # Res=11km as using 0.1deg regridded data for identification (Meteosat res)
            mcs_records.append(make_table(date,hour))
    
    mcs_records=pd.concat(mcs_records)
    domain=args.domain.capitalize()
    outfile = '{}_{}_{}_{}_MCS_table.csv'.format(domain,res_name,config,region.capitalize())
    mcs_records.to_csv(outfile,index=False)

if args.update=="conv_rain":
    get_scheme_rainfall(domain,ffile)
        