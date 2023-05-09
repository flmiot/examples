import re
import numpy as np
import logging
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as cl_random

Log = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG)

### constants ###
cons_C = 299792458
cons_H = 6.62607004e-34 # J*s
cons_E = 1.6021766208e-19
cons_A = 6.022140857e23

def init_state():

    ### OpenCL ###
    platforms = cl.get_platforms()
    devices = [(d.name, pi, di) for pi,p in enumerate(platforms) for di,d in enumerate(p.get_devices())]
    d = {'options':{i[0]: (i[1],i[2]) for i in devices}}
    d['selected'] = [0,0]

    state = {
    "OpenCL":
        {
            "Device":d
        },
    "General":
        {
            "Pump rays":[10000,'int', None],
            "Probe rays":[10000,'int', None],
            "Voxels (^3)":[32,'int', None],
            "Voxel frame":[0.1,'float', None]
        },
    # "View settings":
    #     {
    #         "Pump layout rays":[300,'int', None],
    #         "Probe layout rays":[300,'int', None]
    #     },
    "Pump properties":
        {
            "Pump type":{'options':{"Round gaussian":0, "Rectangular gaussian":1},
                'selected':1},
            "Pump pulse energy":[61.3e-6, 'float', ' J'],
            "Pump wavelength":[400, 'float', ' nm'],
            "Pump radius":[6.0, 'float', ' mm'],
            "Pump width":[6.0, 'float', ' mm'],
            "Pump height":[1.95, 'float', ' mm'],
            "Pump source position":[[0, 79.18, -219.21], 'str', ' mm'],
            "Pump focus":[250, 'float', ' mm']
        },
    "Probe properties":
        {
            "Probe type":{'options':{"Round gaussian":0, "Rectangular gaussian":1},
                'selected':1},
            "Probe radius":[0.05, 'float', ' mm'],
            "Probe width":[0.150, 'float', ' mm'],
            "Probe height":[0.07, 'float', ' mm'],
            "Probe source position":[[0, 0, -50], 'str', ' mm'],
            "Probe focus":[100000, 'float', ' mm'],
            "Solvent abs. coeff":[0.00121, 'float', ' cm^-1']
        },
    "Sample enviroment":
        {
            "Jet type":{'options':{"Round":0, "Rectangular":1},'selected':0},
            "Jet radius":[0.3, 'float', 'mm'],
            "Jet thickness":[0.3, 'float', 'mm'],
            "Jet width":[0.3, 'float', 'mm'],
            "Refr. index 1":[1.0, 'float', None],
            "Refr. index 2":[1.34, 'float', None],
            "Sample concentration":[0.0025, 'float', ' mol L^-1'],
            "Sample att. coeff.":[15675.0, 'float', ' cm^-1'],
            "Solvent concentration":[19.15, 'float', ' mol L^-1'],
            "Solvent att. coeff.":[0.0025, 'float', ' cm^-1']
        }
    }

    return state


def init_results():
    results = {
    "General results":
        {
        "Pump total incident photons":[0, 'int', None],
        "Pump total absorbed photons":[0, 'int', None],
        "Pump mean absorbed photons":[0, 'float', None],
        "Mean weighted excited state fraction":[0, 'float', None],
        "Volume per voxel":[0, 'float', ' L'],
        "Molecules per voxel":[0, 'int', None],
        "Pump dropped rays":[0, 'int', None],
        "Probe dropped rays":[0, 'int', None]
        },
    "Cursor position":
        {
        "Excited state fraction":[0, 'float', None],
        "Weighted excited state fraction":[0, 'float', None],
        "Pump absorbed photons":[0, 'float', None],
        "Probe rel. absorbed photons":[0, 'float', None],
        }
    }
    return results


def check_sim_state(sim_state):
    checked = sim_state
    ints = ["Pump rays","Probe rays","Voxels (^3)", "Pump layout rays",
        "Probe layout rays"]
    arrays = ["Pump source position","Probe source position"]

    for topic in sim_state.keys():
        for property in sim_state[topic]:
            v = sim_state[topic][property]
            if isinstance(v, dict):
                continue
            else:
                value = sim_state[topic][property][0]
                if property in ints:
                    checked[topic][property][0] = int(value)
                elif property in arrays:
                    pattern = r'([-+]?\d*\.\d+|[-+]?\d+)'
                    numbers = re.findall(pattern, str(value))
                    value = list([float(n) for n in numbers])
                    checked[topic][property][0] = value
                else:
                    checked[topic][property][0] = float(value)
    return checked


def read_state_file(filename):
    edit = init_state()
    strings = np.loadtxt(filename, dtype = str, comments='#', delimiter = ":")
    strings = list(zip(*strings))
    for topic_key in edit:
        for property_key in edit[topic_key]:
            ind = strings[0].index(property_key)
            data = edit[topic_key][property_key]
            if isinstance(data, dict):
                    if(property_key == "Device"):
                        pattern = r'(\d+)'
                        numbers = re.findall(pattern, str(strings[1][ind]))
                        value = tuple([int(n) for n in numbers])
                        ind = list(data['options'].values()).index(value)
                        edit[topic_key][property_key]['selected'] = value
                    else:
                        edit[topic_key][property_key]['selected'] = int(strings[1][ind])
            else:
                edit[topic_key][property_key][0] = strings[1][ind]

    Log.info("State file {} read.".format(filename))
    return check_sim_state(edit)


class Simulation:
    # def __init__(self, voxels, voxel_frame, probe_rays, pump_rays, probe_square_source, probe_source_size_x,
    #         probe_source_size_y, probe_source_radius, probe_source_pos, probe_focus, solvent_absorbtion_coef,
    #         pump_square_source, pump_source_size_x, pump_source_size_y, pump_source_radius, pump_source_pos,
    #         pump_focus, jet_radius, n1, n2, pump_pulse_energy, pump_wavelength, sample_concentration,
    #         sample_attcoeff, solvent_concentration, solvent_attcoeff, slice_start, slice_end, layout_rays):
    def __init__(self, platform = 0, device = 0):



        # Results
        self.r = {
            "arrays":{
                # Probe
                "Probe r0":None,
                "Probe r1":None,
                "Probe r2":None,
                "Probe abs.":None,
                # Pump
                "Pump r0":None,
                "Pump r1":None,
                "Pump r2":None,
                "Pump abs.":None,
                "Pump trans.":None,
                "Pump excited state fraction":None,
                "Weighted excited state fraction":None,
            },
            "scalars":{
                "Pump total incident photons":None,
                "Pump total absorbed photons":None,
                "Pump mean absorbed photons":None,
                "Mean weighted excited state fraction":None,
                "Volume per voxel":None,
                "Molecules per voxel":None,
                "Pump dropped rays":None,
                "Probe dropped rays":None
            }
        }


    def run(self, sim_state, call_on_progress):
        """
        Run simulation. Specify **sim_state** dict with simulation parameters,
        and callback function **call_on_progress**
        """

        # import os
        # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

        Log.info("Running: [Step 01] - Setting up")
        progress = 0

        # ==================================================
        # Setup
        # ==================================================
        self.p = {}

        for topic in sim_state.keys():
            for property in sim_state[topic]:
                v = sim_state[topic][property]
                if isinstance(v, dict):
                    self.p[property] = sim_state[topic][property]['selected']
                else:
                    self.p[property] = sim_state[topic][property][0]

        energy_per_photon = cons_H * cons_C / self.p['Pump wavelength'] * 1e9
        photons = self.p['Pump pulse energy'] / energy_per_photon
        photons_per_ray = photons / self.p['Pump rays']
        limit = self.p['Jet radius']+self.p['Jet radius']*self.p['Voxel frame']
        volume_per_voxel_litre = (2*limit/self.p['Voxels (^3)'])**3 * 1e-6
        mol_per_voxel = self.p['Sample concentration'] * volume_per_voxel_litre
        molecules_per_voxel = cons_A * mol_per_voxel

        self.limit = limit

        self.r['scalars']['Pump total incident photons'] = photons
        self.r['scalars']["Volume per voxel"] = volume_per_voxel_litre
        self.r['scalars']["Molecules per voxel"] = molecules_per_voxel


        # ==================================================
        # Init OpenCL & build kernels
        # ==================================================

        platform = self.p['Device'][0]
        device = self.p['Device'][1]
        device = cl.get_platforms()[platform].get_devices()[device]
        ctx = cl.Context([device])
        self.queue = cl.CommandQueue(ctx, device)
        def get_source(filename):
            with open(filename, 'r') as source_file:
                data = source_file.read()
            return data
        src = get_source('gpu/kernels.cl')
        self.kernels = cl.Program(ctx, src).build()

        Log.info("OpenCL successfully initialized for {}".format(device))


        # ==================================================
        # Build a voxel grid
        # ==================================================

        Log.info("Running: [Step 02] - Building voxel grid")
        call_on_progress(10)

        voxels = cl_array.empty(self.queue, self.p['Voxels (^3)']**3, cl_array.vec.float8)
        sides = cl_array.empty(self.queue, self.p['Voxels (^3)']**3, cl_array.vec.int8)
        borders = cl_array.empty(self.queue, self.p['Voxels (^3)']+1, np.float32)
        self.kernels.make_voxel_borders(
            self.queue, borders.shape[::-1], None,
            borders.data, np.float32(-limit), np.float32(limit), np.int32(self.p['Voxels (^3)']+1)
        )

        self.kernels.make_voxels(
            self.queue, voxels.shape[::-1], None,
            voxels.data, sides.data, np.int32(self.p['Voxels (^3)']), borders.data
            )
        vox = self.p['Voxels (^3)']

        Log.info("Running: [Step 03] - Tracing pump rays")
        call_on_progress(20)

        # ==================================================
        # Trace PUMP rays
        # ==================================================

        # Set up ray memory
        r0 = cl_array.empty(self.queue, self.p['Pump rays'], cl_array.vec.float4)
        r1 = cl_array.empty(self.queue, self.p['Pump rays'], cl_array.vec.float4)
        r2 = cl_array.empty(self.queue, self.p['Pump rays'], cl_array.vec.float4)

        if self.p['Pump type'] == 1:
            # h_mem = cl_random.rand(self.queue, self.p['Pump rays'], np.float32, None,
            #     a = -self.p['Pump height']/2, b = self.p['Pump height']/2)
            # w_mem = cl_random.rand(self.queue, self.p['Pump rays'], np.float32, None,
            #     a = -self.p['Pump width']/2, b = self.p['Pump width']/2)
            sx, sy = np.random.rand(1)* 2**32-1, np.random.rand(1)* 2**32-1
            h_mem = cl_array.empty(self.queue, self.p['Pump rays'], np.float32)
            w_mem = cl_array.empty(self.queue, self.p['Pump rays'], np.float32)
            self.kernels.make_square_gaussian_distributed_points(
                self.queue, h_mem.shape[::-1], None, r0.data,
                h_mem.data, w_mem.data, np.uint32(sx), np.uint32(sy), np.float32(self.p['Pump radius']),
                np.float32(self.p['Pump width']), np.float32(self.p['Pump height'])
            )
        else:
            sx, sy = np.random.rand(1) * 2**32-1, np.random.rand(1) * 2**32-1
            h_mem = cl_array.empty(self.queue, self.p['Pump rays'], np.float32)
            w_mem = cl_array.empty(self.queue, self.p['Pump rays'], np.float32)
            self.kernels.make_circular_gaussian_distributed_points(
                self.queue, h_mem.shape[::-1], None, r0.data,
                h_mem.data, w_mem.data, np.uint32(sx), np.uint32(sy), np.float32(self.p['Pump radius'])
            )

        # plt.scatter(w_mem.get(), h_mem.get(), s = 0.1)
        # #plt.hist(h_mem.get(), bins = 1024)
        # plt.show()

        pos = np.array(self.p['Pump source position'])
        t = self.p['Pump focus'] - np.linalg.norm(pos)
        v = -pos/np.linalg.norm(pos) * t
        pump_focal_point = np.array([v[0], v[1], v[2], 0], dtype = np.float32)
        pump_focal_point_mem = cl_array.to_device(self.queue, pump_focal_point)

        counter = cl_array.empty(self.queue,1, np.float32)
        counter.fill(value = np.float32(0.0), queue = self.queue)

        self.kernels.get_photons_per_ray(
            self.queue, r0.shape[::-1], None,
            r0.data, counter.data
        )

        self.r['scalars']['Pump dropped rays'] = int(self.p['Pump rays']-counter.get()[0])
        pump_photons_per_ray = photons/counter.get()[0]

        self.kernels.path_trace_round_jet(
            self.queue, r0.shape[::-1], None,
            r0.data, r1.data, r2.data,
            pump_focal_point_mem.data, np.float32(self.p['Jet radius']),
            cl_array.vec.make_float4(*self.p['Pump source position'],0),
            h_mem.data, w_mem.data,
            np.float32(self.p['Refr. index 1']), np.float32(self.p['Refr. index 2']),
            np.float32(pump_photons_per_ray)
        )

        # Get layout rays
        # l0,l1,l2= self.get_layout_rays(r0.get(), r1.get(),r2.get(), self.p['Pump layout rays'], self.p['Pump layout rays'])
        # self.r['Pump r0'] = l0
        # self.r['Pump r1'] = l1
        # self.r['Pump r2'] = l2

        Log.info("Running: [Step 04] - Rasterize pump rays")
        call_on_progress(60)

        # ==================================================
        # Rasterize PUMP rays
        # ==================================================
        absorbed = cl_array.empty(self.queue, self.p['Voxels (^3)']**3, np.float32)
        transmitted = cl_array.empty(self.queue, self.p['Voxels (^3)']**3, np.float32)
        absorbed.fill(value = np.float32(0.0), queue = self.queue)
        transmitted.fill(value = np.float32(0.0), queue = self.queue)

        sample_abs = self.p["Sample concentration"] * self.p["Sample att. coeff."]
        solvent_abs = self.p["Solvent concentration"]*self.p["Solvent att. coeff."]
        self.kernels.trace_ray_through_voxels(
            self.queue, r1.shape[::-1], None,
            r1.data, r2.data, voxels.data, sides.data, np.int32(self.p['Voxels (^3)']),
            borders.data, absorbed.data, transmitted.data,
            np.float32(sample_abs), np.float32(solvent_abs),
            )

        self.r['arrays']['Pump abs.'] = \
            absorbed.get(queue = self.queue).reshape((vox,vox,vox)).astype(np.float32)

        self.r['arrays']["Pump excited state fraction"] = \
            self.r['arrays']['Pump abs.'] / molecules_per_voxel

        self.r['scalars']["Pump mean absorbed photons"] = \
            np.mean(self.r['arrays']['Pump abs.'])

        self.r['scalars']["Pump total absorbed photons"] = \
            np.sum(self.r['arrays']['Pump abs.'])


        # ==================================================#
        # Trace PROBE rays
        # ==================================================#
        Log.info("Running: [Step 05] - Trace probe rays")
        r0 = cl_array.empty(self.queue, self.p['Probe rays'], cl_array.vec.float4)
        r1 = cl_array.empty(self.queue, self.p['Probe rays'], cl_array.vec.float4)
        r2 = cl_array.empty(self.queue, self.p['Probe rays'], cl_array.vec.float4)

        if self.p['Probe type'] == 1:
            # h_mem = cl_random.rand(self.queue, self.p['Probe rays'], np.float32, None,
            #     a = -self.p['Probe height']/2, b = self.p['Probe height']/2)
            # w_mem = cl_random.rand(self.queue, self.p['Probe rays'], np.float32, None,
            #     a = -self.p['Probe width']/2, b = self.p['Probe width']/2)
            sx, sy = np.random.rand(1) * 2**32-1, np.random.rand(1) * 2**32-1
            h_mem = cl_array.empty(self.queue, self.p['Probe rays'], np.float32)
            w_mem = cl_array.empty(self.queue, self.p['Probe rays'], np.float32)
            self.kernels.make_square_gaussian_distributed_points(
                self.queue, h_mem.shape[::-1], None, r0.data,
                h_mem.data, w_mem.data, np.uint32(sx), np.uint32(sy), np.float32(self.p['Probe radius']),
                np.float32(self.p['Probe width']), np.float32(self.p['Probe height'])
            )
        else:
            sx, sy = np.random.rand(1) * 2**32-1, np.random.rand(1) * 2**32-1
            h_mem = cl_array.empty(self.queue, self.p['Probe rays'], np.float32)
            w_mem = cl_array.empty(self.queue, self.p['Probe rays'], np.float32)
            self.kernels.make_circular_gaussian_distributed_points(
                self.queue, h_mem.shape[::-1], None, r0.data,
                h_mem.data, w_mem.data, np.uint32(sx), np.uint32(sy), np.float32(self.p['Probe radius'])
            )

        # plt.scatter(w_mem.get(), h_mem.get(), s = 0.1)
        # #plt.hist(h_mem.get(), bins = 1024)
        # plt.show()

        pos = np.array(self.p['Probe source position'])
        t = self.p['Probe focus'] - np.linalg.norm(pos)
        v = -pos/np.linalg.norm(pos) * t
        Probe_focal_point = np.array([v[0], v[1], v[2], 0], dtype = np.float32)
        Probe_focal_point_mem = cl_array.to_device(self.queue, Probe_focal_point)

        counter = cl_array.empty(self.queue,1, np.float32)
        counter.fill(value = np.float32(0.0), queue = self.queue)

        self.kernels.get_photons_per_ray(
            self.queue, r0.shape[::-1], None,
            r0.data, counter.data
        )
        self.r['scalars']['Probe dropped rays'] = int(self.p['Probe rays']-counter.get()[0])
        probe_photons_per_ray = photons/counter.get()[0]


        self.kernels.path_trace_round_jet(
            self.queue, r0.shape[::-1], None,
            r0.data, r1.data, r2.data,
            Probe_focal_point_mem.data, np.float32(self.p['Jet radius']),
            cl_array.vec.make_float4(*self.p['Probe source position'],0),
            h_mem.data, w_mem.data,
            np.float32(1), np.float32(1),
            np.float32(probe_photons_per_ray)
        )

        # Get layout rays
        # l0,l1,l2 = self.get_layout_rays(r0.get(), r1.get(),r2.get(), self.p['Probe layout rays'], self.p['Probe layout rays'])
        # self.r['Probe r0'] = l0
        # self.r['Probe r1'] = l1
        # self.r['Probe r2'] = l2

        # ==================================================#
        # Rasterize PROBE rays
        # ==================================================#
        Log.info("Running: [Step 06] - Rasterize probe rays")
        absorbed = cl_array.empty(self.queue, self.p['Voxels (^3)']**3, np.float32)
        transmitted = cl_array.empty(self.queue, self.p['Voxels (^3)']**3, np.float32)
        absorbed.fill(value = np.float32(0.0), queue = self.queue)
        transmitted.fill(value = np.float32(0.0), queue = self.queue)

        self.kernels.trace_ray_through_voxels(
            self.queue, r1.shape[::-1], None,
            r1.data, r2.data, voxels.data, sides.data, np.int32(self.p['Voxels (^3)']),
            borders.data, absorbed.data, transmitted.data,
            np.float32(self.p["Solvent abs. coeff"]), np.float32(0),
            )
        self.r['arrays']['Probe abs.'] = absorbed.get(queue = self.queue).reshape((vox,vox,vox)).astype(np.float32)

        self.r['arrays']["Weighted excited state fraction"] = \
            self.r['arrays']["Pump excited state fraction"] * self.r['arrays']['Probe abs.'] / np.max(self.r['arrays']['Probe abs.'])


        self.r['scalars']["Mean weighted excited state fraction"] = \
            np.mean(self.r['arrays']["Weighted excited state fraction"][self.r['arrays']['Probe abs.'] > 0])



        l1 = np.sum(self.r['arrays']['Probe abs.'] > 0)
        l2 = len(self.r['arrays']['Probe abs.'].ravel())

        non_zero = self.r['arrays']['Probe abs.']/np.max(self.r['arrays']['Probe abs.'])
        non_zero[:] = 0
        non_zero[self.r['arrays']['Probe abs.'] > 0] = 1

        #
        # plt.matshow(non_zero[:,:,16])
        # #plt.matshow(self.r['arrays']["Weighted excited state fraction"] [:,:,16])
        # plt.show()



        Log.info("Running: [END] - OpenCL finished")
        call_on_progress(100)


    def get_layout_rays(self, r0, r1, r2, rays, length):
        r0 = r0[np.arange(0, length).astype(np.int)]
        r1 = r1[np.arange(0, length).astype(np.int)]
        r2 = r2[np.arange(0, length).astype(np.int)]
        non_zero_i = np.where(np.array(list(zip(*r2))[3]) > 0)
        r0 = r0[non_zero_i]
        r1 = r1[non_zero_i]
        r2 = r2[non_zero_i]
        # indizes = []
        # zero_i = np.where(np.array(list(zip(*new_r2)))[3] < 0)
        # print(zero_i)
        # indizes.extend(np.where(np.array(list(zip(*new_r2)))[3] > 0))
        # missing = len(zero_i)
        #
        # for counter in range(int(np.floor(rays/length))):
        #     if missing == 0:
        #         break
        #     b = r2[np.arange(counter*length, counter*length+missing).astype(np.int)]
        #     new_r2[x] = b
        #     zero_i = np.where(np.array(list(zip(*new_r2)))[3] < 0)
        #     indizes.extend(np.where(np.array(list(zip(*new_r2)))[3] > 0))
        #     missing = len(zero_i)
        #     print(zero_i)
        # return r0[np.array(indizes)], r1[np.array(indizes)], new_r2
        return r0,r1,r2
