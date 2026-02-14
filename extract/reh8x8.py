from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import *
import json
import math
import random
from datetime import datetime
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

backwardCompatibility.setValues(includeDeprecated=True, reportDeprecated=False)


class reh:
    def __init__(self, n : int, l : float, h : float, t : float, theta : float, material : dict, n_max : int = None, rxtra : float = None, w : float = None) -> None:
        self.n, self.material, self.n_max, self.rxtra = n, material, n_max, rxtra
        self.theta_deg = 90 - theta
        th = math.radians(theta)
        # self.h = h / 2 - t * math.tan(th) / 2 - t / math.cos(th)
        self.h = h / 2 - t * math.tan(th) / 2 - t / (2 * math.cos(th))
        self.t = t / 2
        self.l = l - t / (2 * math.cos(th))
        self.theta = math.radians(self.theta_deg)
        self.w = w if w is not None else 1.0

        self.sheetsz = 500.0
        self.name = f'l{l:.2f}_h{h:.2f}_t{t:.2f}_theta{theta:.2f}'.replace('.', '-')
        self.cell_side_edge = 2 * (self.h + self.t / math.tan(self.theta) + 2 * self.t / math.sin(self.theta))
        self.cell_height = self.cell_side_edge + 2 * (self.h - self.l * math.cos(self.theta))
        self.cell_length = 2 * (self.l * math.sin(self.theta) + self.t)
        self.length, self.height = self.n * self.cell_length, self.n * self.cell_height

        # print('Cell height, length:', self.cell_height, self.cell_length)
        # print('Lattice height, length:', self.height, self.length)

        self.strain_displacement = 0.01 * self.length
        
        self.disc_rmin = self.h * math.tan(self.theta / 2)
        self.is_reinforced = self.n_max is not None
        if self.is_reinforced:
            self.name += f'_n_max{n_max}_rxtra{rxtra:.2f}'.replace('.', '-')
        self.discs = []

        self.name += datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

        mdb.models.changeKey(fromName='Model-1', toName=self.name)
        self.model = mdb.models[self.name]
        self.sketch = self.model.ConstrainedSketch(name='reh_sketch', sheetSize=self.sheetsz)
        

    def reh_unit(self, x : float, y : float) -> list[float]:
        inner = [
            self.l,
            self.h,
            self.l * math.sin(self.theta), 
            self.h - self.l * math.cos(self.theta),
        ]
        outer = [
            self.t,
            self.h,
            self.l,
            self.cell_side_edge / 2,
            self.cell_length / 2,
            self.cell_height / 2,
        ]
        ret = outer.copy()

        inner = [
            (0, inner[3]),
            (inner[2], inner[1]),
            (inner[2], -inner[1]),
            (0, -inner[3]),
            (-inner[2], -inner[1]),
            (-inner[2], inner[1]),
            (0, inner[3]),
        ]
        outer = [
            (outer[0], outer[5]),
            (outer[0], outer[5] - outer[1]),
            (outer[4], outer[3]),
            (outer[4], -outer[3]),
            (outer[0], outer[1] - outer[5]),
            (outer[0], -outer[5]),
            (-outer[0], -outer[5]),
            (-outer[0], outer[1] - outer[5]),
            (-outer[4], -outer[3]),
            (-outer[4], outer[3]),
            (-outer[0], outer[5] - outer[1]),
            (-outer[0], outer[5]),
            (outer[0], outer[5]),
        ]

        inner, outer = map(lambda ar : [(x + _x, y + _y) for _x, _y in ar], (inner, outer))

        for i in range(len(inner) - 1):
            self.sketch.Line(point1=inner[i], point2=inner[i + 1])
        to_skip = [2, 5, 8, 11]
        for i in range(len(outer) - 1):
            if i in to_skip: continue
            self.sketch.Line(point1=outer[i], point2=outer[i + 1])

        return ret


    def reh_base(self) -> None:
        outer = self.reh_unit(self.cell_length / 2, self.cell_height / 2)
        self.sketch.linearPattern(
            geomList=self.sketch.geometry.values(),
            number1=self.n, spacing1=self.cell_length, angle1=0.0,
            number2=self.n, spacing2=self.cell_height, angle2=90.0,
        )

        for i in range(self.n):
            h = (i + 0.5) * self.cell_height
            self.sketch.Line(point1=(0, h - outer[3]), point2=(0, h + outer[3]))
            self.sketch.Line(point1=(self.length, h - outer[3]), point2=(self.length, h + outer[3]))

        for i in range(self.n):
            l = (i + 0.5) * self.cell_length
            self.sketch.Line(point1=(l - outer[0], 0), point2=(l + outer[0], 0))
            self.sketch.Line(point1=(l - outer[0], self.height), point2=(l + outer[0], self.height))

        self.part = self.model.Part(name='reh_part', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        self.part.BaseShell(sketch=self.sketch)

    
    def reinforce(self, max_tries : int = 1000, fail_cnt_thres : int = 10) -> None:
        if not self.is_reinforced:
            return dict(n = 0, r = 0)

        sketch = self.model.ConstrainedSketch(name='reh_rein_sketch', sheetSize=self.sheetsz)
        r = self.disc_rmin + self.rxtra
        
        fc = 0
        for _ in range(self.n_max):
            for __ in range(max_tries):
                x, y = map(lambda x : random.uniform(r, x - r), (self.length, self.height))
                if all((x - cx)**2 + (y - cy)**2 >= (r + cr)**2 for cx, cy, cr in self.discs):
                    self.discs.append((x, y, r))
                    sketch.CircleByCenterPerimeter((x, y), (x + r, y))
                    break
            else:
                fc += 1
                if fc == fail_cnt_thres: break

        self.rein_part = self.model.Part(name='reh_rein_part', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        self.rein_part.BaseShell(sketch=sketch)

        return dict(n = len(self.discs), r = r)


    def reinforcement_drawing(self) -> None:
        fig, ax = plt.subplots(figsize=(self.length/20, self.height/20), dpi=100)
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        for x, y, r in self.discs:
            ax.add_patch(plt.Circle((x, y), r, color='white'))
        ax.axis('off')
        plt.savefig(f'./dataset/images/{self.name}.png', dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
        # plt.savefig('./sample.png', dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close(fig)

        # self.viewport = session.viewports['Viewport: 1']
        # session.printOptions.setValues(rendition=BLACK_AND_WHITE, vpDecorations=OFF)
        # self.viewport.setValues(displayedObject=self.part)
        # self.viewport.viewportAnnotationOptions.setValues(triad=OFF)
        # # session.printToFile(fileName=f'./dataset/images/{self.name}.png', format=PNG, canvasObjects=(self.viewport,))
        # session.printToFile(fileName='./sample.png', format=PNG, canvasObjects=(self.viewport,))


    def seed_part(self) -> dict:
        self.model.Material(name=self.material['name'])
        self.model.materials[self.material['name']].Elastic(table=((self.material['E'], self.material['v']),))

        if self.is_reinforced:
            self.inst = self.model.rootAssembly.Instance(name='reh_base_assembly', part=self.part, dependent=ON)
            rein_inst = self.model.rootAssembly.Instance(name='reh_rein_assembly', part=self.rein_part, dependent=ON)
            self.part = self.model.rootAssembly.PartFromBooleanMerge(
                name='reh_merged', instances=(self.inst, rein_inst), 
                keepIntersections=FALSE, mergeNodes=ALL,
                nodeMergingTolerance=1e-3, removeDuplicateElements=ON
            )

            del self.model.parts['reh_part'], self.model.parts['reh_rein_part']
            self.model.rootAssembly.suppressFeatures(('reh_base_assembly', 'reh_rein_assembly'))
        
        self.model.HomogeneousSolidSection(name='reh_section', material=self.material['name'], thickness=self.w)
        self.part.SectionAssignment(region=(self.part.faces,), sectionName='reh_section')

        self.quad_meshing_element = mesh.ElemType(elemCode=CPS8R, elemLibrary=STANDARD)
        self.tri_meshing_element = mesh.ElemType(elemCode=CPS6, elemLibrary=STANDARD)
        self.part.setElementType(regions=(self.part.faces,), elemTypes=(self.quad_meshing_element, self.tri_meshing_element))
        self.part.seedPart(size=self.t / 2)
        self.part.generateMesh()

        self.inst = self.model.rootAssembly.Instance(name='reh_assembly', part=self.part, dependent=ON)

        self.volume = self.part.getArea(faces=self.part.faces) * self.w
        return dict(VF = self.volume / (self.height * self.length * self.w))
        

    def tensile_test(self) -> None:
        nodes = dict(left = [], right = [], top = [], bottom = [])
        tol = 1e-3
        for n in self.inst.nodes:
            if abs(n.coordinates[1]) < tol: nodes['bottom'].append(n.label)
            elif abs(n.coordinates[0]) < tol: nodes['left'].append(n.label)
            elif abs(n.coordinates[1] - self.height) < tol: nodes['top'].append(n.label)
            elif abs(n.coordinates[0] - self.length) < tol: nodes['right'].append(n.label)

        regions = {}
        make_region = lambda labels : regionToolset.Region(nodes=self.inst.nodes.sequenceFromLabels(labels=labels))
        for k in ('left', 'right', 'bottom'):
            regions[k] = make_region(nodes[k])
        regions['top'] = [make_region(nodes['top'][:1]), make_region(nodes['top'][1:])]

        self.model.StaticStep(name='Tension', previous='Initial')
        self.model.fieldOutputRequests.changeKey(fromName='F-Output-1', toName='reh_outputs')
        self.model.fieldOutputRequests['reh_outputs'].setValues(variables=('U', 'RF'))
        self.model.DisplacementBC(
            name='u1_left', createStepName='Initial', 
            region=regions['left'], u1=0.0
        )
        self.model.DisplacementBC(
            name='u2_bottom', createStepName='Initial', 
            region=regions['bottom'], u2=0.0
        )
        self.model.DisplacementBC(
            name='u1_right', createStepName='Tension', 
            region=regions['right'], u1=self.strain_displacement
        )
        self.model.Coupling(
            name='u2_top', surface=regions['top'][1],
            controlPoint=regions['top'][0],
            influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
            u1=OFF, u2=ON, u3=OFF, ur1=OFF, ur2=OFF, ur3=OFF
        )

        job = mdb.Job(name=self.name, model=self.name, type=ANALYSIS)
        job.setValues(numCpus=8, numDomains=8, multiprocessingMode=DEFAULT)
        job.submit()
        job.waitForCompletion()

        odb = openOdb(f'{self.name}.odb', readOnly=True)
        U = odb.steps['Tension'].frames[-1].fieldOutputs['U'].values
        RF = odb.steps['Tension'].frames[-1].fieldOutputs['RF'].values
        RF, U = map(lambda res : {v.nodeLabel : v.data for v in res}, (RF, U))
        odb.close()

        # print('Left side sum of reaction force:', np.sum([RF[n][0] for n in nodes['left']]))
        # print([RF[n][0] for n in nodes['left']])

        eps_xx = self.strain_displacement / self.length
        eps_yy = np.mean([U[n][1] for n in nodes['top']]) / self.height
        sigma_xx = -np.sum([RF[n][0] for n in nodes['left']]) / (self.height * self.w)

        return dict(experiment = self.name, E = sigma_xx / eps_xx, v = -eps_yy / eps_xx)

    
    def simulate(base_params : dict, rein_params : dict = {}) -> dict:
        structure = __class__(**base_params, **rein_params)
        structure.reh_base()
        res = structure.reinforce()
        res |= structure.seed_part()
        res |= structure.tensile_test()
        structure.reinforcement_drawing()
        res['frac_E'] = res['E'] / structure.material['E']
        return res


if __name__ == '__main__':
    with open('./dataset/results.csv', 'a', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=['experiment', 'E', 'v', 'n', 'r', 'VF', 'frac_E'])
        base_params = dict(n = 8, l = 42, t = 3, h = 42, theta = 23, material = dict(name = 'steel', E = 210e3, v = 0.3))
        with open('config.json', 'r') as fp:
            rein_params = json.load(fp)
        csv_writer.writerow(reh.simulate(base_params, rein_params))

    # base_params = dict(n = 8, l = 42, t = 3, h = 42, theta = 23, material = dict(name = 'steel', E = 210e3, v = 0.3), w = 7.74)
    # rein_params = dict(n_max = 42, rxtra = 1.2)
    # # rein_params = {}
    # print(reh.simulate(base_params, rein_params))