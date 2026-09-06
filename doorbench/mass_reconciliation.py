"""Apply per-panel material/hardware budgets without counting geometry twice."""
from __future__ import annotations


def reconcile_moving_mass(model, phys):
    leaves={b.name:b for b in model.bodies if b.semantic=='leaf' and not b.static}
    rows={row['body']:row for row in phys['mass']['per_body']}
    if set(leaves)!=set(rows):
        raise ValueError(f'{model.name}: physical mass panel names differ from IR leaves: {sorted(rows)} vs {sorted(leaves)}')
    if not leaves:return
    primary=next(iter(leaves));bodies={b.name:b for b in model.bodies};hardware={name:[] for name in leaves}
    backed=set(model.meta.get('mechanism_mass_bodies', []))
    if any(n not in bodies or bodies[n].static or n in leaves for n in backed):
        raise ValueError('mechanism_mass_bodies must name moving non-leaf bodies')
    transfers=model.meta.get('material_transfer_bodies',{})
    if not isinstance(transfers,dict) or any(not isinstance(n,str) or not isinstance(owner,str) for n,owner in transfers.items()):
        raise ValueError('material_transfer_bodies must map body names to source leaf names')
    if any(n not in bodies or bodies[n].static or n in leaves or owner not in leaves for n,owner in transfers.items()):
        raise ValueError('Material transfers require moving non-leaf bodies and existing source leaves')
    if backed.intersection(transfers):
        raise ValueError('Material transfers cannot also add a mechanism mass budget')
    # Repeated reconciliation must neither add the same BOM twice nor rescale
    # a mechanism which is already backed by its authored material geometry.
    previous=float(phys['mass']['hardware_parts'].pop('geometry_backed_mechanisms',0.))
    phys['mass']['hardware_kg']-=previous;phys['mass']['total_kg']-=previous
    for row in rows.values():
        previous=float(row['hardware_parts'].pop('geometry_backed_mechanisms',0.))
        row['hardware_kg']-=previous;row['total_kg']-=previous
    for body in model.bodies:
        if body.static or body.name in leaves:continue
        parent=body.parent;seen=set()
        while parent and parent not in leaves and parent not in seen:
            seen.add(parent);parent=bodies[parent].parent
        # World-mounted moving closer arms/carriers belong to the primary
        # assembly. Their stationary housings are excluded by body.static.
        owner=parent if parent in leaves else primary
        if body.name in transfers and (parent!=transfers[body.name] or owner!=transfers[body.name]):
            raise ValueError('Material transfer source must be the moving body ancestor leaf')
        hardware[owner].append(body)
    report=[]
    for name,leaf in leaves.items():
        row=rows[name];physical=[b for b in hardware[name] if b.name in backed]
        transferred={b.name:float(b.inertial('full')[0]) for b in hardware[name] if b.name in transfers}
        if any(not (m>0) for m in transferred.values()):
            raise ValueError('Material transfers require positive authored mass')
        material_total=float(row['slab_kg']+row['glass_kg']);transferred_total=sum(transferred.values())
        if transferred_total>material_total+1e-9:
            raise ValueError('Material transfer exceeds the source leaf material budget')
        parts=[b for b in hardware[name] if b.name not in backed and b.name not in transfers]
        physical_masses={b.name:float(b.inertial('full')[0]) for b in physical}
        if any(m<=0 for m in physical_masses.values()):
            raise ValueError('Geometry-backed mechanisms require positive authored mass')
        physical_total=sum(physical_masses.values())
        raw=[max(float(b.inertial('full')[0]),1e-9) for b in parts]
        raw_total=sum(raw);budget=float(row['hardware_kg'])
        # Every articulated link needs positive inertia. If a newly authored
        # mechanism has no catalogue allowance, expose the tiny reserve rather
        # than silently deleting its mass or reducing the slab below material.
        floor=.001*len(parts)
        reserve=max(0.,floor-budget)
        if reserve:
            row['hardware_parts']['modeled_link_minimum_reserve']=reserve
            row['hardware_kg']+=reserve;row['total_kg']+=reserve
            phys['mass']['hardware_kg']+=reserve;phys['mass']['total_kg']+=reserve
            phys['mass']['hardware_parts']['modeled_link_minimum_reserve']=phys['mass']['hardware_parts'].get('modeled_link_minimum_reserve',0.)+reserve
            budget+=reserve
        carried=min(raw_total,budget)
        scale=carried/raw_total if raw_total else 1.
        for body,mass in zip(parts,raw):body.mass_override=mass*scale
        leaf.mass_override=float(material_total-transferred_total+budget-carried)
        if physical_total:
            row['hardware_parts']['geometry_backed_mechanisms']=physical_total
            row['hardware_kg']+=physical_total;row['total_kg']+=physical_total
            phys['mass']['hardware_kg']+=physical_total;phys['mass']['total_kg']+=physical_total
            hp=phys['mass']['hardware_parts']
            hp['geometry_backed_mechanisms']=hp.get('geometry_backed_mechanisms',0.)+physical_total
        report.append({'body':name,'material_kg':material_total,'leaf_body_kg':leaf.mass_override,'hardware_budget_kg':budget,'separate_hardware_kg':carried,'hardware_calibration_scale':scale,'separate_hardware_bodies':[b.name for b in parts], 'geometry_backed_bodies_kg':physical_masses,'geometry_backed_kg':physical_total})
        if transfers:report[-1].update(transferred_material_bodies_kg=transferred,transferred_material_kg=transferred_total)
    total=sum(float(b.inertial('full')[0]) for b in model.bodies if not b.static)
    if abs(total-phys['mass']['total_kg'])>1e-7:
        raise ValueError(f'{model.name}: native moving mass {total} differs from assembly budget {phys["mass"]["total_kg"]}')
    model.meta['mass_reconciled_kg']=sum(b.mass_override for b in leaves.values())
    model.meta['moving_assembly_mass_kg']=total
    phys['mass']['geometry_backed_mechanisms_kg']=sum(r['geometry_backed_kg'] for r in report)
    if transfers:phys['mass']['transferred_material_kg']=sum(r['transferred_material_kg'] for r in report)
    phys['mass']['slab_and_catalogue_hardware_budget_kg']=total-phys['mass']['geometry_backed_mechanisms_kg']
    phys['mass']['dynamics_model_limitation']=('Analytical per-panel friction, inertia and counterbalance sizing use the '
        'slab-and-catalogue hardware budget. Explicit articulated mechanism masses are added to the native assembly; '
        'their generalized gravity and force contributions require the native linkage model, not a rigid-panel multiplier.')
    model.meta['mass_reconciliation']={'scope':'per-panel material, catalogue hardware, plus explicit mechanism BOM','panels':report,'moving_total_kg':total,'limitations':'Proxy hardware uses catalogue budgets. Explicit mechanism_mass_bodies retain authored inertia and add to assembly mass; no strength certification.'}
    if transfers:model.meta['mass_reconciliation']['limitations']+=' material_transfer_bodies allocate existing material to articulated children without adding it twice; the caller must match the declared construction.'
