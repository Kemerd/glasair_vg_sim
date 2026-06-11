#include "setup.hpp"



#ifdef BENCHMARK
#include "info.hpp"
void main_setup() { // benchmark; required extensions in defines.hpp: BENCHMARK, optionally FP16S or FP16C
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	uint mlups = 0u; {

		//LBM lbm( 32u,  32u,  32u, 1.0f);
		//LBM lbm( 64u,  64u,  64u, 1.0f);
		//LBM lbm(128u, 128u, 128u, 1.0f);
		LBM lbm(256u, 256u, 256u, 1.0f); // default
		//LBM lbm(384u, 384u, 384u, 1.0f);
		//LBM lbm(512u, 512u, 512u, 1.0f);

		//const uint memory = 1488u; // memory occupation in MB (for multi-GPU benchmarks: make this close to as large as the GPU's VRAM capacity)
		//const uint3 lbm_N = (resolution(float3(1.0f, 1.0f, 1.0f), memory)/4u)*4u; // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
		//LBM lbm(1u*lbm_N.x, 1u*lbm_N.y, 1u*lbm_N.z, 1u, 1u, 1u, 1.0f); // 1 GPU
		//LBM lbm(2u*lbm_N.x, 1u*lbm_N.y, 1u*lbm_N.z, 2u, 1u, 1u, 1.0f); // 2 GPUs
		//LBM lbm(2u*lbm_N.x, 2u*lbm_N.y, 1u*lbm_N.z, 2u, 2u, 1u, 1.0f); // 4 GPUs
		//LBM lbm(2u*lbm_N.x, 2u*lbm_N.y, 2u*lbm_N.z, 2u, 2u, 2u, 1.0f); // 8 GPUs

		// #########################################################################################################################################################################################
		for(uint i=0u; i<1000u; i++) {
			lbm.run(10u, 1000u*10u);
			mlups = max(mlups, to_uint((double)lbm.get_N()*1E-6/info.runtime_lbm_timestep_smooth));
		}
	} // make lbm object go out of scope to free its memory
	print_info("Peak MLUPs/s = "+to_string(mlups));
#if defined(_WIN32)
	wait();
#endif // Windows
} /**/
#endif // BENCHMARK

#ifndef BENCHMARK
#include <fstream>
#include <sstream>
#include <map>
// Glasair III VG study tunnel -- config-file driven so AoA / Re / geometry /
// domain mode change WITHOUT recompiling. Config format: "key=value" lines,
// '#' comments, at L:/Dev/glasair_vg_sim/gpu/fluidx3d/tunnel_config.txt
// (overridable as the first command-line argument).
// required extensions in defines.hpp: FP16S, FORCE_FIELD, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS
static std::map<std::string, std::string> read_config(const std::string& path) {
	std::map<std::string, std::string> cfg;
	std::ifstream f(path);
	std::string line;
	while(std::getline(f, line)) {
		const size_t hash = line.find('#');
		if(hash!=std::string::npos) line = line.substr(0, hash);
		const size_t eq = line.find('=');
		if(eq==std::string::npos) continue;
		auto trim = [](std::string s) {
			const size_t a = s.find_first_not_of(" \t\r");
			const size_t b = s.find_last_not_of(" \t\r");
			return a==std::string::npos ? std::string() : s.substr(a, b-a+1u);
		};
		cfg[trim(line.substr(0, eq))] = trim(line.substr(eq+1u));
	}
	return cfg;
}
static float cfgf(std::map<std::string, std::string>& c, const std::string& k, const float d) {
	return c.count(k) ? (float)atof(c[k].c_str()) : d;
}
static std::string cfgs(std::map<std::string, std::string>& c, const std::string& k, const std::string& d) {
	return c.count(k) ? c[k] : d;
}

void main_setup() { // Glasair III VG-study tunnel (LS(1)-0413 section, factory-measured chord)
	// ################################################################## configuration ###################################################################
	// NOTE: FluidX3D's stock main() interprets command-line arguments as GPU
	// device IDs, so the config location is fixed instead of argument-passed:
	// tunnel.bat writes per-run overrides to tunnel_run.txt; a bare
	// double-click of the exe falls back to the editable tunnel_config.txt.
	std::string cfg_path = "L:/Dev/glasair_vg_sim/gpu/fluidx3d/tunnel_run.txt";
	if(!std::ifstream(cfg_path).good()) cfg_path = "L:/Dev/glasair_vg_sim/gpu/fluidx3d/tunnel_config.txt";
	std::map<std::string, std::string> cfg = read_config(cfg_path);
	print_info("config: "+cfg_path);
	const std::string stl_path = cfgs(cfg, "stl", "L:/Dev/glasair_vg_sim/gpu/fluidx3d/assets/wing_a0_binary.stl");
	const std::string mode    = cfgs(cfg, "mode", "wing");   // wing = whole section + tips | slice = narrow high-res strip, span-periodic
	const std::string csv     = cfgs(cfg, "csv", "");        // force log path; empty = no CSV
	const float aoa_deg  = cfgf(cfg, "aoa_deg", 8.0f);
	const float lbm_Re   = cfgf(cfg, "re", 1.0e6f);          // LES-effective Reynolds number (the "airspeed" knob: Re = V*c/nu)
	const float vram_mb  = cfgf(cfg, "vram_mb", 6000.0f);
	const float lbm_u    = cfgf(cfg, "u", 0.10f);            // lattice inflow speed (keep ~0.05..0.12 for stability)
	const float si_span  = cfgf(cfg, "span_m", 1.5f);        // STL true span in meters
	const float si_chord = cfgf(cfg, "chord_m", 0.9022f);    // STL true chord in meters [DXF measured]
	const ulong log_every= (ulong)cfgf(cfg, "log_every", 1000.0f);
	const float t_end_si = cfgf(cfg, "t_end_si", 0.0f);      // seconds of PHYSICAL airflow to simulate; 0 = ignore
	const float video_s  = cfgf(cfg, "video_s", 0.0f);       // export a 360-deg orbit as PNG frames spanning this many seconds of 60 fps video; 0 = off
	const std::string frames_dir = cfgs(cfg, "frames_dir", "L:/Dev/glasair_vg_sim/gpu/fluidx3d/results/frames/");
	ulong t_end = (ulong)cfgf(cfg, "t_end_steps", 0.0f);     // 0 = run forever (interactive)
	// ################################################################## box + scaling ###################################################################
	const bool slice = (mode=="slice");
	// wing mode: span along x with free tips inside the box. slice mode: the
	// strip pierces BOTH periodic x faces -- the span is scaled slightly
	// PAST the box (overhang, default 2%) so the physical tips sit outside
	// the simulated volume entirely: an infinite wing with no edge/tip
	// artifacts at the walls (owner-specified; exact wall-to-wall scaling
	// left residual edge vortices).
	const float overhang = cfgf(cfg, "span_overhang", 1.02f);
	const float aspect_x = slice ? (si_span/si_chord) : 1.0f;
	const uint3 lbm_N = resolution(slice ? float3(aspect_x, 2.2f, 1.2f) : float3(1.0f, 2.0f, 0.6f), (uint)vram_mb);
	const float lbm_span  = slice ? overhang*(float)lbm_N.x : 0.55f*(float)lbm_N.x;
	const float lbm_chord = lbm_span*(si_chord/si_span);
	const float lbm_scale = fmaxf(lbm_span, lbm_chord);      // read_stl scales the mesh's LONGEST side to this
	LBM lbm(lbm_N, units.nu_from_Re(lbm_Re, lbm_chord, lbm_u));
	// Physical-time bookkeeping: one lattice step represents dt_si seconds of
	// real airflow (u_lat maps onto the Re-implied physical airspeed at the
	// true chord). Lets the config say "simulate 1.2 seconds of air" directly.
	const float u_si  = lbm_Re*1.4607e-5f/si_chord;          // physical airspeed implied by Re at this chord, ISA sea level
	const float dt_si = lbm_u*(si_chord/lbm_chord)/u_si;     // seconds of physics per lattice step
	if(t_end_si>0.0f) t_end = (ulong)(t_end_si/dt_si);
	print_info("VG tunnel: mode="+mode+", aoa="+to_string(aoa_deg, 1u)+" deg, Re="+to_string(lbm_Re)
	           +", chord="+to_string(lbm_chord, 1u)+" cells ("+to_string(si_chord/lbm_chord*1000.0f, 2u)+" mm/cell)");
	print_info("physics: V = "+to_string(u_si, 1u)+" m/s ("+to_string(u_si*2.23694f, 1u)+" mph), dt = "
	           +to_string(dt_si*1.0e6f, 2u)+" us/step"+(t_end>0ull ? ", auto-close after "+to_string(t_end)+" steps = "
	           +to_string(t_end_si, 2u)+" s of airflow" : ", running until closed"));
	// ################################################################## geometry ###################################################################
	// STL frame: x=chord, y=thickness, z=span. LBM frame: flow +y, up +z, span +x.
	Mesh* mesh = read_stl(stl_path, lbm.size(), lbm.center(), float3x3(float3(1, 0, 0), radians(90.0f)), lbm_scale);
	mesh->rotate(float3x3(float3(0, 0, 1), radians(90.0f)));     // span onto +x, thickness up onto +z
	mesh->rotate(float3x3(float3(1, 0, 0), radians(-aoa_deg)));  // pitch LE up by the angle of attack
	mesh->translate(float3(0.0f, -0.15f*lbm.size().y, 0.0f));    // park upstream so the wake can develop
	lbm.voxelize_mesh_on_device(mesh, TYPE_S|TYPE_X);            // TYPE_X marks the wing for object_force()
	// ---------------------------------------------------------------------
	// Analytic VG stamping. The STL voxelizer parity-counts along the SPAN
	// (its minimum-cross-section heuristic) and one grazing miss on a thin
	// vane plate erases every vane after it (owner-observed half-row bug).
	// Vanes are therefore stamped directly into the lattice from their
	// parametric definition: exact at any resolution, immune to parity, and
	// VG sweeps become pure config changes (no STL regeneration).
	// Frame bookkeeping: the wing mesh is bbox-centered at lbm.center(),
	// parked 0.15*Ny upstream, and pitched -aoa about the span axis; cells
	// are tested in the un-pitched wing frame (chord +y, up +z, span +x).
	const bool vg_on   = cfgf(cfg, "vg_enable", 0.0f)!=0.0f;
	const float vg_h   = cfgf(cfg, "vg_h_mm", 10.0f)/1000.0f/si_chord*lbm_chord;   // vane height, lattice units
	const float vg_p   = cfgf(cfg, "vg_pitch_mm", 50.0f)/1000.0f/si_chord*lbm_chord;
	const float vg_t   = cfgf(cfg, "vg_t_mm", 6.0f)/1000.0f/si_chord*lbm_chord;    // plate thickness (6 mm visual default)
	const float vg_l   = cfgf(cfg, "vg_l_per_h", 3.0f)*vg_h;
	const float vg_b   = radians(cfgf(cfg, "vg_beta_deg", 15.0f));
	const float z_skin = cfgf(cfg, "vg_skin_off_frac", 0.03149f)*lbm_chord;  // upper skin at 7%c vs mesh bbox center
	const float y_row  = cfgf(cfg, "vg_le_off_frac", -0.43f)*lbm_chord;      // row LE vs mesh bbox center (chordwise)
	const float ar     = radians(-aoa_deg);                      // pitch rotation the mesh received
	const float3 wc    = lbm.center()+float3(0.0f, -0.15f*lbm.size().y, 0.0f); // wing bbox center in the box
	// Speck (coherence breaker): ONE small bump stamped at mid-span, 30%c --
	// breaks the artificial spanwise-roller coherence of the quasi-2D domain
	// so the visible flow behaves like honest 3D air (user-validated method).
	const bool speck_on  = cfgf(cfg, "speck_enable", 0.0f)!=0.0f;
	const float sp_h     = cfgf(cfg, "speck_h_mm", 8.0f)/1000.0f/si_chord*lbm_chord;
	const float sp_l     = cfgf(cfg, "speck_l_mm", 8.0f)/1000.0f/si_chord*lbm_chord;
	const float sp_w     = cfgf(cfg, "speck_w_mm", 3.0f)/1000.0f/si_chord*lbm_chord;
	const float sp_skin  = cfgf(cfg, "speck_skin_off_frac", 0.06208f)*lbm_chord; // upper skin at 30%c vs bbox center
	const float sp_le    = cfgf(cfg, "speck_le_off_frac", -0.20f)*lbm_chord;
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(speck_on&&lbm.flags[n]==0u) {                         // stamp the mid-span speck
			const float dx=(float)x-wc.x, dy=(float)y-wc.y, dz=(float)z-wc.z;
			const float yw =  dy*cos(ar)+dz*sin(ar);
			const float zw = -dy*sin(ar)+dz*cos(ar);
			// Minimum one-cell footprint in every direction: a physically
			// thin solid must never fall between lattice cell centers and
			// silently vanish (the sub-voxel dropout class of bug).
			const float w2 = fmax(0.5f*sp_w, 0.75f);
			const float l2 = fmax(sp_l, 1.5f), h2 = fmax(sp_h, 1.5f);
			if(fabs(dx)<w2&&zw>sp_skin-1.5f&&zw<sp_skin+h2
			   &&yw>sp_le&&yw<sp_le+l2) lbm.flags[n] = TYPE_S|TYPE_X;
		}
		if(vg_on&&lbm.flags[n]==0u) {                            // stamp vanes into still-fluid cells
			const float dx=(float)x-wc.x, dy=(float)y-wc.y, dz=(float)z-wc.z;
			const float yw =  dy*cos(ar)+dz*sin(ar);             // un-pitch into the wing frame
			const float zw = -dy*sin(ar)+dz*cos(ar);
			if(zw>z_skin-1.5f&&zw<z_skin+vg_h) {                 // vertical band: skin (with sink) to vane tip
				const float dyv = yw-y_row;                      // chordwise distance from the row leading edge
				if(dyv>-vg_t&&dyv<vg_l+vg_t) {
					const float ul = fmod(fmod(dx, vg_p)+vg_p, vg_p); // span position inside the local pair cell
					for(int side=0; side<2; side++) {            // two toe-out plates per pair
						const float sgn = side==0 ? 1.0f : -1.0f;
						const float du = ul-(side==0 ? 0.25f : 0.75f)*vg_p;
						const float along  = du*sin(sgn*vg_b)+dyv*cos(vg_b); // plate-frame coordinates
						const float across = du*cos(vg_b)-dyv*sin(sgn*vg_b);
						// Same one-cell floor as the speck: physical 1.5 mm
						// plates must still exist on a coarse visual lattice.
						if(along>=0.0f&&along<=vg_l&&fabs(across)<=fmax(0.5f*vg_t, 0.75f)) { lbm.flags[n] = TYPE_S|TYPE_X; break; }
					}
				}
			}
		}
		if(lbm.flags[n]!=(TYPE_S|TYPE_X)&&lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		const bool xwall = (x==0u||x==Nx-1u) && !slice;          // slice mode: x faces stay periodic (infinite VG array)
		if(xwall||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E;
	}); // ################################################################## run + measure ###################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	// Startup camera framing (owner request): 2x zoom, 35 deg elevation looking
	// down onto the wing's upper surface. All four knobs live in the config;
	// mouse/keys still take over live, and the film orbit reuses zoom/elevation.
	const float cam_zoom = cfgf(cfg, "cam_zoom", 2.0f);          // bigger = closer (1.0 = stock framing)
	const float cam_ry   = cfgf(cfg, "cam_ry", 35.0f);           // elevation: positive looks down on the wing top
	const float cam_rx   = cfgf(cfg, "cam_rx", 0.0f);            // initial azimuth (autorotation sweeps this)
	const float cam_fov  = cfgf(cfg, "cam_fov", 60.0f);          // field of view
	lbm.graphics.set_camera_centered(cam_rx, cam_ry, cam_fov, cam_zoom);
#ifdef INTERACTIVE_GRAPHICS
	key_P = true;                                                // auto-start: skip the press-P-to-run pause (P still toggles)
	if(cfgf(cfg, "autorotate", 0.0f)!=0.0f) camera.autorotation = true; // optional: launch with the built-in R-key autorotation already on
#endif // INTERACTIVE_GRAPHICS
	const float A_ref = lbm_chord*lbm_span;                      // planform reference area in lattice units
	const float q_ref = 0.5f*lbm_u*lbm_u;                        // lattice dynamic pressure (rho = 1)
	if(!csv.empty()) write_file(csv, "# Glasair VG tunnel force log | mode="+mode+" stl="+stl_path
		+" aoa_deg="+to_string(aoa_deg, 1u)+" Re="+to_string(lbm_Re)+" mm_per_cell="+to_string(si_chord/lbm_chord*1000.0f, 3u)
		+"\n# t,Fx,Fy,Fz,Cd,Cl\n");
	lbm.run(0u);                                                 // initialize
	const bool film = video_s>0.0f && t_end>0ull;                // orbit export needs a finite end time to pace frames against
	uint frame = 0u;
	const uint total_frames = max(1u, (uint)(60.0f*video_s));    // 60 fps video
	auto log_forces = [&]() {
		if(csv.empty()) return;
		const float3 F = lbm.object_force(TYPE_S|TYPE_X);        // integrated lattice force on the wing
		const float Cd = F.y/(q_ref*A_ref);                      // streamwise (+y = flow) -> drag coefficient
		const float Cl = F.z/(q_ref*A_ref);                      // vertical (+z) -> lift coefficient
		write_line(csv, to_string(lbm.get_t())+","+to_string(F.x, 6u)+","+to_string(F.y, 6u)+","
		           +to_string(F.z, 6u)+","+to_string(Cd, 6u)+","+to_string(Cl, 6u)+"\n");
	};
	if(film) {                                                   // filming run: single-step so the frame pacer can fire precisely
		// No scripted orbit: the built-in camera autorotation provides the
		// rotation (scripting the azimuth here FIGHTS autorotation and shows
		// up as a rotate-snap-rotate glitch -- owner-diagnosed). Film mode
		// just ensures autorotation is on and saves the paced frames.
#ifdef INTERACTIVE_GRAPHICS
		camera.autorotation = true;
#endif // INTERACTIVE_GRAPHICS
		print_info("filming: "+to_string(total_frames)+" frames ("+to_string(video_s, 1u)+" s at 60 fps, built-in autorotation) -> "+frames_dir);
		while(lbm.get_t()<t_end) {
			if(lbm.graphics.next_frame(t_end, video_s)) {        // true when it is time to capture the next video frame
				lbm.graphics.write_frame(frames_dir);
				frame++;
			}
			lbm.run(1u, t_end);
			if(lbm.get_t()%log_every==0ull) log_forces();
		}
	} else {                                                     // measurement run (Ahmed-body idiom): chunked stepping
		// Passing the true total as run()'s second argument makes the on-screen
		// gauge show OVERALL progress percent and OVERALL time remaining across
		// chunks (Info::append keeps the running average when a total is given).
		// An endless run must pass max_ulong -- the display then switches to
		// "Elapsed Time"; passing 0 corrupts both gauges (the bug we had).
		while(t_end==0ull||lbm.get_t()<t_end) {
			lbm.run(log_every, t_end>0ull ? t_end : max_ulong);
			log_forces();
		}
	}
	lbm.write_status();
	if(t_end>0ull) print_info("run complete: "+to_string(t_end_si, 2u)+" s of physical airflow simulated"
	                          +(film ? ", "+to_string(frame)+" frames written" : "")+". Closing.");
#ifdef INTERACTIVE_GRAPHICS
	// Finite runs under INTERACTIVE_GRAPHICS crash with an access violation in
	// teardown (the render thread races LBM destruction once main_setup
	// returns; observed as exit code 0xC0000005 with all results intact).
	// All outputs above are already flushed to disk, so end the process
	// deliberately before the broken teardown path can run.
	if(t_end>0ull) exit(0);
#endif // INTERACTIVE_GRAPHICS
} /**/
#endif // !BENCHMARK



/*void main_setup() { // 3D Taylor-Green vortices; required extensions in defines.hpp: INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(128u, 128u, 128u, 1u, 1u, 1u, 0.01f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		const float A = 0.25f;
		const uint periodicity = 1u;
		const float a=(float)Nx/(float)periodicity, b=(float)Ny/(float)periodicity, c=(float)Nz/(float)periodicity;
		const float fx = (float)x+0.5f-0.5f*(float)Nx;
		const float fy = (float)y+0.5f-0.5f*(float)Ny;
		const float fz = (float)z+0.5f-0.5f*(float)Nz;
		lbm.u.x[n] =  A*cosf(2.0f*pif*fx/a)*sinf(2.0f*pif*fy/b)*sinf(2.0f*pif*fz/c);
		lbm.u.y[n] = -A*sinf(2.0f*pif*fx/a)*cosf(2.0f*pif*fy/b)*sinf(2.0f*pif*fz/c);
		lbm.u.z[n] =  A*sinf(2.0f*pif*fx/a)*sinf(2.0f*pif*fy/b)*cosf(2.0f*pif*fz/c);
		lbm.rho[n] = 1.0f-sq(A)*3.0f/4.0f*(cosf(4.0f*pif*fx/a)+cosf(4.0f*pif*fy/b));
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_STREAMLINES;
	lbm.run();
	//lbm.run(1000u); lbm.u.read_from_device(); println(lbm.u.x[lbm.index(Nx/2u, Ny/2u, Nz/2u)]); wait(); // test for binary identity
} /**/



/*void main_setup() { // 2D Taylor-Green vortices (use D2Q9); required extensions in defines.hpp: INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(1024u, 1024u, 1u, 0.02f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		const float A = 0.2f;
		const uint periodicity = 5u;
		const float a=(float)Nx/(float)periodicity, b=(float)Ny/(float)periodicity;
		const float fx = (float)x+0.5f-0.5f*(float)Nx;
		const float fy = (float)y+0.5f-0.5f*(float)Ny;
		lbm.u.x[n] =  A*cosf(2.0f*pif*fx/a)*sinf(2.0f*pif*fy/b);
		lbm.u.y[n] = -A*sinf(2.0f*pif*fx/a)*cosf(2.0f*pif*fy/b);
		lbm.rho[n] = 1.0f-sq(A)*3.0f/4.0f*(cosf(4.0f*pif*fx/a)+cosf(4.0f*pif*fy/b));
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FIELD;
	lbm.graphics.slice_mode = 3;
	lbm.run();
} /**/



/*void main_setup() { // Poiseuille flow validation; required extensions in defines.hpp: VOLUME_FORCE
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint R = 63u; // channel radius (default: 63)
	const float umax = 0.1f; // maximum velocity in channel center (must be < 0.57735027f)
	const float tau = 1.0f; // relaxation time (must be > 0.5f), tau = nu*3+0.5
	const float nu = units.nu_from_tau(tau); // nu = (tau-0.5)/3
	const uint H = 2u*(R+1u);
#ifndef D2Q9
	LBM lbm(H, lcm(sq(H), WORKGROUP_SIZE)/sq(H), H, nu, 0.0f, units.f_from_u_Poiseuille_3D(umax, 1.0f, nu, R), 0.0f); // 3D
#else // D2Q9
	LBM lbm(lcm(H, WORKGROUP_SIZE)/H, H, 1u, nu, units.f_from_u_Poiseuille_2D(umax, 1.0f, nu, R), 0.0f, 0.0f); // 2D
#endif // D2Q9
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
#ifndef D2Q9
		if(!cylinder(x, y, z, lbm.center(), float3(0u, Ny, 0u), 0.5f*(float)min(Nx, Nz)-1.0f)) lbm.flags[n] = TYPE_S; // 3D
#else // D2Q9
		if(y==0u||y==Ny-1u) lbm.flags[n] = TYPE_S; // 2D
#endif // D2Q9
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	double error_min = max_double;
	while(true) { // main simulation loop
		lbm.run(1000u);
		lbm.u.read_from_device();
		double error_dif=0.0, error_sum=0.0;
#ifndef D2Q9
		for(uint x=0u; x<Nx; x++) {
			for(uint y=Ny/2u; y<Ny/2u+1u; y++) {
				for(uint z=0; z<Nz; z++) {
					const uint n = x+(y+z*Ny)*Nx;
					const double r = (double)sqrt(sq(x+0.5f-0.5f*(float)Nx)+sq(z+0.5f-0.5f*(float)Nz)); // radius from channel center
					if(r<R) {
						const double unum = (double)sqrt(sq(lbm.u.x[n])+sq(lbm.u.y[n])+sq(lbm.u.z[n])); // numerical velocity
						const double uref = umax*(sq(R)-sq(r))/sq(R); // theoretical velocity profile u = G*(R^2-r^2)
						error_dif += sq(unum-uref); // L2 error (Krüger p. 138)
						error_sum += sq(uref);
					}
				}
			}
		}
#else // D2Q9
		for(uint x=Nx/2u; x<Nx/2u+1u; x++) {
			for(uint y=1u; y<Ny-1u; y++) {
				const uint n = x+(y+0u*Ny)*Nx;
				const double r = (double)(y+0.5f-0.5f*(float)Ny); // radius from channel center
				const double unum = (double)sqrt(sq(lbm.u.x[n])+sq(lbm.u.y[n])); // numerical velocity
				const double uref = umax*(sq(R)-sq(r))/sq(R); // theoretical velocity profile u = G*(R^2-r^2)
				error_dif += sq(unum-uref); // L2 error (Krüger p. 138)
				error_sum += sq(uref);
			}
		}
#endif // D2Q9
		if(sqrt(error_dif/error_sum)>=error_min) { // stop when error has converged
			print_info("Poiseuille flow error converged after "+to_string(lbm.get_t())+" steps to "+to_string(100.0*error_min, 3u)+"%"); // typical expected L2 errors: 2-5% (Krüger p. 256)
			wait();
			exit(0);
		}
		error_min = fmin(error_min, sqrt(error_dif/error_sum));
		print_info("Poiseuille flow error after t="+to_string(lbm.get_t())+" is "+to_string(100.0*error_min, 3u)+"%"); // typical expected L2 errors: 2-5% (Krüger p. 256)
	}
} /**/



/*void main_setup() { // Stokes drag validation; required extensions in defines.hpp: FORCE_FIELD, EQUILIBRIUM_BOUNDARIES
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const ulong dt = 100ull; // check error every dt time steps
	const float R = 32.0f; // sphere radius
	const float Re = 0.01f; // Reynolds number
	const float nu = 1.0f; // kinematic shear viscosity
	const float rho = 1.0f; // density
	const uint L = to_uint(8.0f*R); // simulation box size
	const float u = units.u_from_Re(Re, 2.0f*R, nu); // velocity
	LBM lbm(L, L, L, nu); // flow driven by equilibrium boundaries
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E;
		if(sphere(x, y, z, lbm.center(), R)) {
			lbm.flags[n] = TYPE_S|TYPE_X; // flag boundary cells for force summation additionally with TYPE_X
		} else {
			lbm.rho[n] = units.rho_Stokes(lbm.position(x, y, z), float3(-u, 0.0f, 0.0f), R, rho, nu);
			const float3 un = units.u_Stokes(lbm.position(x, y, z), float3(-u, 0.0f, 0.0f), R);
			lbm.u.x[n] = un.x;
			lbm.u.y[n] = un.y;
			lbm.u.z[n] = un.z;
		}
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	double E1=1000.0, E2=1000.0;
	while(true) { // main simulation loop
		lbm.run(dt);
		const float3 force = lbm.object_force(TYPE_S|TYPE_X);
		const double F_theo = units.F_Stokes(rho, u, nu, R);
		const double F_sim = (double)length(force);
		const double E0 = fabs(F_sim-F_theo)/F_theo;
		print_info(to_string(lbm.get_t())+", expected: "+to_string(F_theo, 6u)+", measured: "+to_string(F_sim, 6u)+", error = "+to_string((float)(100.0*E0), 1u)+"%");
		if(converged(E2, E1, E0, 1E-4)) { // stop when error has sufficiently converged
			print_info("Error converged after "+to_string(lbm.get_t())+" steps to "+to_string(100.0*E0, 1u)+"%");
			wait();
			break;
		}
		E2 = E1;
		E1 = E0;
	}
} /**/



/*void main_setup() { // cylinder in rectangular duct; required extensions in defines.hpp: VOLUME_FORCE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const float Re = 25000.0f;
	const float D = 64.0f;
	const float u = rsqrt(3.0f);
	const float w=D, l=12.0f*D, h=3.0f*D;
	const float nu = units.nu_from_Re(Re, D, u);
	const float f = units.f_from_u_rectangular_duct(w, D, 1.0f, nu, u);
	LBM lbm(to_uint(w), to_uint(l), to_uint(h), nu, 0.0f, f, 0.0f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		lbm.u.y[n] = 0.1f*u;
		if(cylinder(x, y, z, float3(lbm.center().x, 2.0f*D, lbm.center().z), float3(Nx, 0u, 0u), 0.5f*D)) lbm.flags[n] = TYPE_S;
		if(x==0u||x==Nx-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // x and z non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_Q_CRITERION;
	lbm.run();
} /**/



/*void main_setup() { // Taylor-Couette flow; required extensions in defines.hpp: MOVING_BOUNDARIES, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(96u, 96u, 192u, 1u, 1u, 1u, 0.04f);
	// ###################################################################################### define geometry ######################################################################################
	const uint threads = (uint)thread::hardware_concurrency();
	vector<uint> seed(threads);
	for(uint t=0u; t<threads; t++) seed[t] = 42u+t;
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), threads, [&](ulong n, uint t) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(!cylinder(x, y, z, lbm.center(), float3(0u, 0u, Nz), (float)(Nx/2u-1u))) lbm.flags[n] = TYPE_S;
		if( cylinder(x, y, z, lbm.center(), float3(0u, 0u, Nz), (float)(Nx/4u   ))) {
			const float3 relative_position = lbm.relative_position(n);
			lbm.u.x[n] =  relative_position.y;
			lbm.u.y[n] = -relative_position.x;
			lbm.u.z[n] = (1.0f-random(seed[t], 2.0f))*0.001f;
			lbm.flags[n] = TYPE_S;
		}
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_STREAMLINES;
	lbm.run();
	//lbm.run(4000u); lbm.u.read_from_device(); println(lbm.u.x[lbm.index(Nx/4u, Ny/4u, Nz/2u)]); wait(); // test for binary identity
} /**/



/*void main_setup() { // lid-driven cavity; required extensions in defines.hpp: MOVING_BOUNDARIES, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint L = 128u;
	const float Re = 1000.0f;
	const float u = 0.1f;
	LBM lbm(L, L, L, units.nu_from_Re(Re, (float)(L-2u), u));
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(z==Nz-1) lbm.u.y[n] = u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_STREAMLINES;
	lbm.run();
} /**/



/*void main_setup() { // 2D Karman vortex street; required extensions in defines.hpp: D2Q9, FP16S, EQUILIBRIUM_BOUNDARIES, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint R = 16u;
	const float Re = 250.0f;
	const float u = 0.10f;
	LBM lbm(16u*R, 32u*R, 1u, units.nu_from_Re(Re, 2.0f*(float)R, u));
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(cylinder(x, y, z, float3(Nx/2u, Ny/4u, Nz/2u), float3(0u, 0u, Nz), (float)R)) lbm.flags[n] = TYPE_S;
		else lbm.u.y[n] = u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_FIELD;
	lbm.graphics.slice_mode = 3;
	lbm.run();
} /**/



/*void main_setup() { // particle test; required extensions in defines.hpp: VOLUME_FORCE, FORCE_FIELD, MOVING_BOUNDARIES, PARTICLES, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint L = 128u;
	const float Re = 1000.0f;
	const float u = 0.1f;
	LBM lbm(L, L, L, units.nu_from_Re(Re, (float)(L-2u), u), 0.0f, 0.0f, -0.00001f, cb(L/4u), 2.0f);
	// ###################################################################################### define geometry ######################################################################################
	uint seed = 42u;
	for(ulong n=0ull; n<lbm.particles->length(); n++) {
		lbm.particles->x[n] = random_symmetric(seed, 0.5f*lbm.size().x/4.0f);
		lbm.particles->y[n] = random_symmetric(seed, 0.5f*lbm.size().y/4.0f);
		lbm.particles->z[n] = random_symmetric(seed, 0.5f*lbm.size().z/4.0f);
	}
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(z==Nz-1) lbm.u.y[n] = u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_STREAMLINES|VIS_PARTICLES;
	lbm.run();
} /**/



/*void main_setup() { // delta wing; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint L = 128u;
	const float Re = 100000.0f;
	const float u = 0.075f;
	LBM lbm(L, 4u*L, L, units.nu_from_Re(Re, (float)L, u));
	// ###################################################################################### define geometry ######################################################################################
	const float3 offset = float3(lbm.center().x, 0.0f, lbm.center().z);
	const float3 p0 = offset+float3(  0*(int)L/64,  5*(int)L/64,  20*(int)L/64);
	const float3 p1 = offset+float3(-20*(int)L/64, 90*(int)L/64, -10*(int)L/64);
	const float3 p2 = offset+float3(+20*(int)L/64, 90*(int)L/64, -10*(int)L/64);
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(triangle(x, y, z, p0, p1, p2)) lbm.flags[n] = TYPE_S;
		else lbm.u.y[n] = u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	lbm.run();
} /**/



/*void main_setup() { // NASA Common Research Model; required extensions in defines.hpp: FP16C, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 1.5f, 1.0f/3.0f), 2000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float Re = 10000000.0f;
	const float u = 0.075f;
	LBM lbm(lbm_N, units.nu_from_Re(Re, (float)lbm_N.x, u));
	// ###################################################################################### define geometry ######################################################################################
	// model: https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/assembled-geometry/, .stp file converted to .stl with https://imagetostl.com/convert/file/stp/to/stl
	Mesh* half = read_stl(get_exe_path()+"../stl/crm-hl_reference_ldg.stl", lbm.size(), float3(0.0f), float3x3(float3(0, 0, 1), radians(90.0f)), 1.0f*lbm_N.x);
	half->translate(float3(-0.5f*(half->pmax.x-half->pmin.x), 0.0f, 0.0f));
	Mesh* mesh = new Mesh(2u*half->triangle_number, float3(0.0f));
	for(uint i=0u; i<half->triangle_number; i++) {
		mesh->p0[i] = half->p0[i];
		mesh->p1[i] = half->p1[i];
		mesh->p2[i] = half->p2[i];
	}
	half->rotate(float3x3(float3(1, 0, 0), radians(180.0f))); // mirror-copy half
	for(uint i=0u; i<half->triangle_number; i++) {
		mesh->p0[half->triangle_number+i] = -half->p0[i];
		mesh->p1[half->triangle_number+i] = -half->p1[i];
		mesh->p2[half->triangle_number+i] = -half->p2[i];
	}
	delete half;
	mesh->find_bounds();
	mesh->rotate(float3x3(float3(1, 0, 0), radians(-10.0f)));
	mesh->translate(float3(0.0f, 0.0f, -0.5f*(mesh->pmin.z+mesh->pmax.z)));
	mesh->translate(float3(0.0f, -0.5f*lbm.size().y+mesh->pmax.y+0.5f*(lbm.size().x-(mesh->pmax.x-mesh->pmin.x)), 0.0f));
	mesh->translate(lbm.center());
	lbm.voxelize_mesh_on_device(mesh);
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	lbm.run();
} /**/



/*void main_setup() { // Concorde; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 3.0f, 0.5f), 2084u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float si_u = 300.0f/3.6f;
	const float si_length=62.0f, si_width=26.0f;
	const float si_T = 1.0f;
	const float si_nu=1.48E-5f, si_rho=1.225f;
	const float lbm_length = 0.56f*(float)lbm_N.y;
	const float lbm_u = 0.075f;
	units.set_m_kg_s(lbm_length, lbm_u, 1.0f, si_length, si_u, si_rho);
	const float lbm_nu = units.nu(si_nu);
	const ulong lbm_T = units.t(si_T);
	print_info("Re = "+to_string(to_uint(units.si_Re(si_width, si_u, si_nu))));
	LBM lbm(lbm_N, 1u, 1u, 1u, lbm_nu);
	// ###################################################################################### define geometry ######################################################################################
	const float3 center = float3(lbm.center().x, 0.52f*lbm_length, lbm.center().z+0.03f*lbm_length);
	const float3x3 rotation = float3x3(float3(1, 0, 0), radians(-10.0f))*float3x3(float3(0, 0, 1), radians(90.0f))*float3x3(float3(1, 0, 0), radians(90.0f));
	lbm.voxelize_stl(get_exe_path()+"../stl/concord_cut_large.stl", center, rotation, lbm_length); // https://www.thingiverse.com/thing:1176931/files
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	lbm.run(0u, lbm_T); // initialize simulation
	lbm.write_status();
	while(lbm.get_t()<=lbm_T) { // main simulation loop
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
		if(lbm.graphics.next_frame(lbm_T, 10.0f)) {
			lbm.graphics.set_camera_free(float3(0.491343f*(float)Nx, -0.882147f*(float)Ny, 0.564339f*(float)Nz), -78.0f, 6.0f, 22.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/front/");
			lbm.graphics.set_camera_free(float3(1.133361f*(float)Nx, 1.407077f*(float)Ny, 1.684411f*(float)Nz), 72.0f, 12.0f, 20.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/back/");
			lbm.graphics.set_camera_centered(0.0f, 0.0f, 25.0f, 1.648722f);
			lbm.graphics.write_frame(get_exe_path()+"export/side/");
			lbm.graphics.set_camera_centered(0.0f, 90.0f, 25.0f, 1.648722f);
			lbm.graphics.write_frame(get_exe_path()+"export/top/");
			lbm.graphics.set_camera_free(float3(0.269361f*(float)Nx, -0.179720f*(float)Ny, 0.304988f*(float)Nz), -56.0f, 31.6f, 100.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/wing/");
			lbm.graphics.set_camera_free(float3(0.204399f*(float)Nx, 0.340055f*(float)Ny, 1.620902f*(float)Nz), 80.0f, 35.6f, 34.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/follow/");
		}
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
		lbm.run(1u, lbm_T); // run dt time steps
	}
	lbm.write_status();
} /**/



/*void main_setup() { // Boeing 747; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 2.0f, 0.5f), 880u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_Re = 1000000.0f;
	const float lbm_u = 0.075f;
	const ulong lbm_T = 10000ull;
	LBM lbm(lbm_N, units.nu_from_Re(lbm_Re, (float)lbm_N.x, lbm_u));
	// ###################################################################################### define geometry ######################################################################################
	const float size = 1.0f*lbm.size().x;
	const float3 center = float3(lbm.center().x, 0.55f*size, lbm.center().z);
	const float3x3 rotation = float3x3(float3(1, 0, 0), radians(-15.0f));
	lbm.voxelize_stl(get_exe_path()+"../stl/techtris_airplane.stl", center, rotation, size); // https://www.thingiverse.com/thing:2772812/files
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
	lbm.graphics.set_camera_free(float3(1.0f*(float)Nx, -0.4f*(float)Ny, 2.0f*(float)Nz), -33.0f, 42.0f, 68.0f);
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<lbm_T) { // main simulation loop
		if(lbm.graphics.next_frame(lbm_T, 10.0f)) lbm.graphics.write_frame(); // render enough frames 10 seconds of 60fps video
		lbm.run(1u, lbm_T);
	}
#else // GRAPHICS && !INTERACTIVE_GRAPHICS
	lbm.run();
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
} /**/



/*void main_setup() { // Star Wars X-wing; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 2.0f, 0.5f), 880u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_Re = 100000.0f;
	const float lbm_u = 0.075f;
	const ulong lbm_T = 50000ull;
	LBM lbm(lbm_N, units.nu_from_Re(lbm_Re, (float)lbm_N.x, lbm_u));
	// ###################################################################################### define geometry ######################################################################################
	const float size = 1.0f*lbm.size().x;
	const float3 center = float3(lbm.center().x, 0.55f*size, lbm.center().z);
	const float3x3 rotation = float3x3(float3(0, 0, 1), radians(180.0f));
	lbm.voxelize_stl(get_exe_path()+"../stl/X-Wing.stl", center, rotation, size); // https://www.thingiverse.com/thing:353276/files
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<lbm_T) { // main simulation loop
		if(lbm.graphics.next_frame(lbm_T, 30.0f)) {
			lbm.graphics.set_camera_free(float3(1.0f*(float)Nx, -0.4f*(float)Ny, 2.0f*(float)Nz), -33.0f, 42.0f, 68.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/t/");
			lbm.graphics.set_camera_free(float3(0.5f*(float)Nx, -0.35f*(float)Ny, -0.7f*(float)Nz), -33.0f, -40.0f, 100.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/b/");
			lbm.graphics.set_camera_free(float3(0.0f*(float)Nx, 0.51f*(float)Ny, 0.75f*(float)Nz), 90.0f, 28.0f, 80.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/f/");
			lbm.graphics.set_camera_free(float3(0.7f*(float)Nx, -0.15f*(float)Ny, 0.06f*(float)Nz), 0.0f, 0.0f, 100.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/s/");
		}
		lbm.run(1u, lbm_T);
	}
#else // GRAPHICS && !INTERACTIVE_GRAPHICS
	lbm.run();
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
} /**/



/*void main_setup() { // Star Wars TIE fighter; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 2.0f, 1.0f), 1760u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_Re = 100000.0f;
	const float lbm_u = 0.075f;
	const ulong lbm_T = 50000ull;
	const ulong lbm_dt = 28ull;
	LBM lbm(lbm_N, units.nu_from_Re(lbm_Re, (float)lbm_N.x, lbm_u));
	// ###################################################################################### define geometry ######################################################################################
	const float size = 0.65f*lbm.size().x;
	const float3 center = float3(lbm.center().x, 0.6f*size, lbm.center().z);
	const float3x3 rotation = float3x3(float3(1, 0, 0), radians(90.0f));
	Mesh* mesh = read_stl(get_exe_path()+"../stl/DWG_Tie_Fighter_Assembled_02.stl", lbm.size(), center, rotation, size); // https://www.thingiverse.com/thing:2919109/files
	lbm.voxelize_mesh_on_device(mesh);
	lbm.flags.read_from_device();
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<lbm_T) { // main simulation loop
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
		if(lbm.graphics.next_frame(lbm_T, 30.0f)) {
			lbm.graphics.set_camera_free(float3(1.0f*(float)Nx, -0.4f*(float)Ny, 0.63f*(float)Nz), -33.0f, 33.0f, 80.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/t/");
			lbm.graphics.set_camera_free(float3(0.3f*(float)Nx, -1.5f*(float)Ny, -0.45f*(float)Nz), -83.0f, -10.0f, 40.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/b/");
			lbm.graphics.set_camera_free(float3(0.0f*(float)Nx, 0.57f*(float)Ny, 0.7f*(float)Nz), 90.0f, 29.5f, 80.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/f/");
			lbm.graphics.set_camera_free(float3(2.5f*(float)Nx, 0.0f*(float)Ny, 0.0f*(float)Nz), 0.0f, 0.0f, 50.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/s/");
		}
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
		lbm.run(lbm_dt, lbm_T);
		const float3x3 rotation = float3x3(float3(0.2f, 1.0f, 0.1f), radians(0.4032f)); // create rotation matrix to rotate mesh
		lbm.unvoxelize_mesh_on_device(mesh);
		mesh->rotate(rotation); // rotate mesh
		lbm.voxelize_mesh_on_device(mesh);
	}
} /**/



/*void main_setup() { // radial fan; required extensions in defines.hpp: FP16S, MOVING_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(3.0f, 3.0f, 1.0f), 181u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_Re = 100000.0f;
	const float lbm_u = 0.1f;
	const ulong lbm_T = 48000ull;
	const ulong lbm_dt = 10ull;
	LBM lbm(lbm_N, 1u, 1u, 1u, units.nu_from_Re(lbm_Re, (float)lbm_N.x, lbm_u));
	// ###################################################################################### define geometry ######################################################################################
	const float radius = 0.25f*(float)lbm_N.x;
	const float3 center = float3(lbm.center().x, lbm.center().y, 0.36f*radius);
	const float lbm_omega=lbm_u/radius, lbm_domega=lbm_omega*lbm_dt;
	Mesh* mesh = read_stl(get_exe_path()+"../stl/FAN_Solid_Bottom.stl", lbm.size(), center, 2.0f*radius); // https://www.thingiverse.com/thing:6113/files
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u) lbm.flags[n] = TYPE_S; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<lbm_T) { // main simulation loop
		lbm.voxelize_mesh_on_device(mesh, TYPE_S, center, float3(0.0f), float3(0.0f, 0.0f, lbm_omega));
		lbm.run(lbm_dt, lbm_T);
		mesh->rotate(float3x3(float3(0.0f, 0.0f, 1.0f), lbm_domega)); // rotate mesh
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
		if(lbm.graphics.next_frame(lbm_T, 30.0f)) {
			lbm.graphics.set_camera_free(float3(0.353512f*(float)Nx, -0.150326f*(float)Ny, 1.643939f*(float)Nz), -25.0f, 61.0f, 100.0f);
			lbm.graphics.write_frame();
		}
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
	}
} /**/



/*void main_setup() { // electric ducted fan (EDF); required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, MOVING_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 1.5f, 1.0f), 8000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_Re = 1000000.0f;
	const float lbm_u = 0.1f;
	const ulong lbm_T = 180000ull;
	const ulong lbm_dt = 4ull;
	LBM lbm(lbm_N, units.nu_from_Re(lbm_Re, (float)lbm_N.x, lbm_u));
	// ###################################################################################### define geometry ######################################################################################
	const float3 center = lbm.center();
	const float3x3 rotation = float3x3(float3(0, 0, 1), radians(180.0f));
	Mesh* stator = read_stl(get_exe_path()+"../stl/edf_v39.stl", 1.0f, rotation); // https://www.thingiverse.com/thing:3014759/files
	Mesh* rotor = read_stl(get_exe_path()+"../stl/edf_v391.stl", 1.0f, rotation); // https://www.thingiverse.com/thing:3014759/files
	const float scale = 0.98f*stator->get_scale_for_box_fit(lbm.size()); // scale stator and rotor to simulation box size
	stator->scale(scale);
	rotor->scale(scale);
	stator->translate(lbm.center()-stator->get_bounding_box_center()-float3(0.0f, 0.2f*stator->get_max_size(), 0.0f)); // move stator and rotor to simulation box center
	rotor->translate(lbm.center()-rotor->get_bounding_box_center()-float3(0.0f, 0.41f*stator->get_max_size(), 0.0f));
	stator->set_center(stator->get_center_of_mass()); // set rotation center of mesh to its center of mass
	rotor->set_center(rotor->get_center_of_mass());
	const float lbm_radius=0.5f*rotor->get_max_size(), omega=lbm_u/lbm_radius, domega=omega*(float)lbm_dt;
	lbm.voxelize_mesh_on_device(stator, TYPE_S, center);
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]==0u) lbm.u.y[n] = 0.3f*lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<lbm_T) { // main simulation loop
		lbm.voxelize_mesh_on_device(rotor, TYPE_S, center, float3(0.0f), float3(0.0f, omega, 0.0f));
		lbm.run(lbm_dt, lbm_T);
		rotor->rotate(float3x3(float3(0.0f, 1.0f, 0.0f), domega)); // rotate mesh
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
		if(lbm.graphics.next_frame(lbm_T, 30.0f)) {
			lbm.graphics.set_camera_centered(-70.0f+100.0f*(float)lbm.get_t()/(float)lbm_T, 2.0f, 60.0f, 1.284025f);
			lbm.graphics.write_frame();
		}
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
	}
} /**/



/*void main_setup() { // aerodynamics of a cow; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 2.0f, 1.0f), 1000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float si_u = 1.0f;
	const float si_length = 2.4f;
	const float si_T = 10.0f;
	const float si_nu=1.48E-5f, si_rho=1.225f;
	const float lbm_length = 0.65f*(float)lbm_N.y;
	const float lbm_u = 0.075f;
	units.set_m_kg_s(lbm_length, lbm_u, 1.0f, si_length, si_u, si_rho);
	const float lbm_nu = units.nu(si_nu);
	const ulong lbm_T = units.t(si_T);
	print_info("Re = "+to_string(to_uint(units.si_Re(si_length, si_u, si_nu))));
	LBM lbm(lbm_N, lbm_nu);
	// ###################################################################################### define geometry ######################################################################################
	const float3x3 rotation = float3x3(float3(1, 0, 0), radians(180.0f))*float3x3(float3(0, 0, 1), radians(180.0f));
	Mesh* mesh = read_stl(get_exe_path()+"../stl/Cow_t.stl", lbm.size(), lbm.center(), rotation, lbm_length); // https://www.thingiverse.com/thing:182114/files
	mesh->translate(float3(0.0f, 1.0f-mesh->pmin.y+0.1f*lbm_length, 1.0f-mesh->pmin.z)); // move mesh forward a bit and to simulation box bottom, keep in mind 1 cell thick box boundaries
	lbm.voxelize_mesh_on_device(mesh);
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(z==0u) lbm.flags[n] = TYPE_S; // solid floor
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u; // initialize y-velocity everywhere except in solid cells
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all other simulation box boundaries are inflow/outflow
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
	lbm.graphics.set_camera_centered(-40.0f, 20.0f, 78.0f, 1.25f);
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<=lbm_T) { // main simulation loop
		if(lbm.graphics.next_frame(lbm_T, 10.0f)) lbm.graphics.write_frame();
		lbm.run(1u, lbm_T);
	}
#else // GRAPHICS && !INTERACTIVE_GRAPHICS
	lbm.run();
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
} /**/



/*void main_setup() { // Space Shuttle; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 4.0f, 0.8f), 1000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_Re = 10000000.0f;
	const float lbm_u = 0.075f;
	const ulong lbm_T = 108000ull;
	LBM lbm(lbm_N, 2u, 4u, 1u, units.nu_from_Re(lbm_Re, (float)lbm_N.x, lbm_u)); // run on 2x4x1 = 8 GPUs
	// ###################################################################################### define geometry ######################################################################################
	const float size = 1.25f*lbm.size().x;
	const float3 center = float3(lbm.center().x, 0.55f*size, lbm.center().z+0.05f*size);
	const float3x3 rotation = float3x3(float3(1, 0, 0), radians(-20.0f))*float3x3(float3(0, 0, 1), radians(270.0f));
	Clock clock;
	lbm.voxelize_stl(get_exe_path()+"../stl/Full_Shuttle.stl", center, rotation, size); // https://www.thingiverse.com/thing:4975964/files
	println(print_time(clock.stop()));
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_FLAG_SURFACE|VIS_Q_CRITERION;
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
	lbm.write_status();
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<=lbm_T) { // main simulation loop
		if(lbm.graphics.next_frame(lbm_T, 30.0f)) {
			lbm.graphics.set_camera_free(float3(-1.435962f*(float)Nx, 0.364331f*(float)Ny, 1.344426f*(float)Nz), -205.0f, 36.0f, 74.0f); // top
			lbm.graphics.write_frame(get_exe_path()+"export/top/");
			lbm.graphics.set_camera_free(float3(-1.021207f*(float)Nx, -0.518006f*(float)Ny, 0.0f*(float)Nz), -137.0f, 0.0f, 74.0f); // bottom
			lbm.graphics.write_frame(get_exe_path()+"export/bottom/");
		}
		lbm.run(1u, lbm_T);
	}
	lbm.write_status();
#else // GRAPHICS && !INTERACTIVE_GRAPHICS
	lbm.run();
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
} /**/



/*void main_setup() { // Starship; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 2.0f, 2.0f), 1000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_Re = 10000000.0f;
	const float lbm_u = 0.05f;
	const ulong lbm_T = 108000ull;
	LBM lbm(lbm_N, 1u, 1u, 1u, units.nu_from_Re(lbm_Re, (float)lbm_N.x, lbm_u));
	// ###################################################################################### define geometry ######################################################################################
	const float size = 1.6f*lbm.size().x;
	const float3 center = float3(lbm.center().x, lbm.center().y+0.05f*size, 0.18f*size);
	lbm.voxelize_stl(get_exe_path()+"../stl/StarShipV2.stl", center, size); // https://www.thingiverse.com/thing:4912729/files
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.z[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_FLAG_SURFACE|VIS_Q_CRITERION;
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
	lbm.write_status();
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<=lbm_T) { // main simulation loop
		if(lbm.graphics.next_frame(lbm_T, 20.0f)) {
			lbm.graphics.set_camera_free(float3(2.116744f*(float)Nx, -0.775261f*(float)Ny, 1.026577f*(float)Nz), -38.0f, 37.0f, 60.0f); // top
			lbm.graphics.write_frame(get_exe_path()+"export/top/");
			lbm.graphics.set_camera_free(float3(0.718942f*(float)Nx, 0.311263f*(float)Ny, -0.498366f*(float)Nz), 32.0f, -40.0f, 104.0f); // bottom
			lbm.graphics.write_frame(get_exe_path()+"export/bottom/");
			lbm.graphics.set_camera_free(float3(1.748119f*(float)Nx, 0.442782f*(float)Ny, 0.087945f*(float)Nz), 24.0f, 2.0f, 92.0f); // side
			lbm.graphics.write_frame(get_exe_path()+"export/side/");
		}
		lbm.run(1u, lbm_T);
	}
	lbm.write_status();
#else // GRAPHICS && !INTERACTIVE_GRAPHICS
	lbm.run();
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
} /**/



/*void main_setup() { // Ahmed body; required extensions in defines.hpp: FP16C, FORCE_FIELD, EQUILIBRIUM_BOUNDARIES, SUBGRID, optionally INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint memory = 10000u; // available VRAM of GPU(s) in MB
	const float lbm_u = 0.05f;
	const float box_scale = 6.0f;
	const float si_u = 60.0f;
	const float si_nu=1.48E-5f, si_rho=1.225f;
	const float si_width=0.389f, si_height=0.288f, si_length=1.044f;
	const float si_A = si_width*si_height+2.0f*0.05f*0.03f;
	const float si_T = 0.25f;
	const float si_Lx = units.x(box_scale*si_width);
	const float si_Ly = units.x(box_scale*si_length);
	const float si_Lz = units.x(0.5f*(box_scale-1.0f)*si_width+si_height);
	const uint3 lbm_N = resolution(float3(si_Lx, si_Ly, si_Lz), memory); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	units.set_m_kg_s((float)lbm_N.y, lbm_u, 1.0f, box_scale*si_length, si_u, si_rho);
	const float lbm_nu = units.nu(si_nu);
	const ulong lbm_T = units.t(si_T);
	const float lbm_length = units.x(si_length);
	print_info("Re = "+to_string(to_uint(units.si_Re(si_width, si_u, si_nu))));
	LBM lbm(lbm_N, lbm_nu);
	// ###################################################################################### define geometry ######################################################################################
	Mesh* mesh = read_stl(get_exe_path()+"../stl/ahmed_25deg_m.stl", lbm.size(), lbm.center(), float3x3(float3(0, 0, 1), radians(90.0f)), lbm_length);
	mesh->translate(float3(0.0f, units.x(0.5f*(0.5f*box_scale*si_length-si_width))-mesh->pmin.y, 1.0f-mesh->pmin.z));
	lbm.voxelize_mesh_on_device(mesh, TYPE_S|TYPE_X); // https://github.com/nathanrooy/ahmed-bluff-body-cfd/blob/master/geometry/ahmed_25deg_m.stl converted to binary
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(z==0u) lbm.flags[n] = TYPE_S;
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==Nz-1u) lbm.flags[n] = TYPE_E;
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_FIELD;
	lbm.graphics.field_mode = 1;
	lbm.graphics.slice_mode = 1;
	//lbm.graphics.set_camera_centered(20.0f, 30.0f, 10.0f, 1.648722f);
	lbm.run(0u, lbm_T); // initialize simulation
#if defined(FP16S)
	const string path = get_exe_path()+"FP16S/"+to_string(memory)+"MB/";
#elif defined(FP16C)
	const string path = get_exe_path()+"FP16C/"+to_string(memory)+"MB/";
#else // FP32
	const string path = get_exe_path()+"FP32/"+to_string(memory)+"MB/";
#endif // FP32
	//lbm.write_status(path);
	//write_file(path+"Cd.dat", "# t\tCd\n");
	const float3 lbm_com = lbm.object_center_of_mass(TYPE_S|TYPE_X);
	print_info("com = "+to_string(lbm_com.x, 2u)+", "+to_string(lbm_com.y, 2u)+", "+to_string(lbm_com.z, 2u));
	while(lbm.get_t()<=lbm_T) { // main simulation loop
		Clock clock;
		const float3 lbm_force = lbm.object_force(TYPE_S|TYPE_X);
		//const float3 lbm_torque = lbm.object_torque(lbm_com, TYPE_S|TYPE_X);
		//print_info("F="+to_string(lbm_force.x, 2u)+","+to_string(lbm_force.y, 2u)+","+to_string(lbm_force.z, 2u)+", T="+to_string(lbm_torque.x, 2u)+","+to_string(lbm_torque.y, 2u)+","+to_string(lbm_torque.z, 2u)+", t="+to_string(clock.stop(), 3u));
		const float Cd = units.si_F(lbm_force.y)/(0.5f*si_rho*sq(si_u)*si_A); // expect Cd to be too large by a factor 1.3-2.0x; need wall model
		print_info("Cd = "+to_string(Cd, 3u)+", t = "+to_string(clock.stop(), 3u));
		//write_line(path+"Cd.dat", to_string(lbm.get_t())+"\t"+to_string(Cd, 3u)+"\n");
		lbm.run(1u, lbm_T);
	}
	//lbm.write_status(path);
} /**/



/*void main_setup() { // Cessna 172 propeller aircraft; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, MOVING_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 0.8f, 0.25f), 8000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_u = 0.075f;
	const float lbm_width = 0.95f*(float)lbm_N.x;
	const ulong lbm_dt = 4ull; // revoxelize rotor every dt time steps
	const float si_T = 1.0f;
	const float si_width = 11.0f;
	const float si_u = 226.0f/3.6f;
	const float si_nu=1.48E-5f, si_rho=1.225f;
	units.set_m_kg_s(lbm_width, lbm_u, 1.0f, si_width, si_u, si_rho);
	const float lbm_nu = units.nu(si_nu);
	const ulong lbm_T = units.t(si_T);
	print_info("Re = "+to_string(to_uint(units.si_Re(si_width, si_u, si_nu))));
	print_info(to_string(si_T, 3u)+" seconds = "+to_string(lbm_T)+" time steps");
	LBM lbm(lbm_N, units.nu(si_nu));
	// ###################################################################################### define geometry ######################################################################################
	Mesh* plane = read_stl(get_exe_path()+"../stl/Cessna-172-Skyhawk-body.stl"); // https://www.thingiverse.com/thing:814319/files
	Mesh* rotor = read_stl(get_exe_path()+"../stl/Cessna-172-Skyhawk-rotor.stl"); // plane and rotor separated with Microsoft 3D Builder
	const float scale = lbm_width/plane->get_bounding_box_size().x; // scale plane and rotor to simulation box size
	plane->scale(scale);
	rotor->scale(scale);
	const float3 offset = lbm.center()-plane->get_bounding_box_center(); // move plane and rotor to simulation box center
	plane->translate(offset);
	rotor->translate(offset);
	plane->set_center(plane->get_center_of_mass()); // set rotation center of mesh to its center of mass
	rotor->set_center(rotor->get_center_of_mass());
	const float lbm_radius=0.5f*rotor->get_max_size(), omega=-lbm_u/lbm_radius, domega=omega*(float)lbm_dt;
	lbm.voxelize_mesh_on_device(plane);
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<=lbm_T) { // main simulation loop
		lbm.voxelize_mesh_on_device(rotor, TYPE_S, rotor->get_center(), float3(0.0f), float3(0.0f, omega, 0.0f)); // revoxelize mesh on GPU
		lbm.run(lbm_dt, lbm_T); // run dt time steps
		rotor->rotate(float3x3(float3(0.0f, 1.0f, 0.0f), domega)); // rotate mesh
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
		if(lbm.graphics.next_frame(lbm_T, 5.0f)) {
			lbm.graphics.set_camera_free(float3(0.192778f*(float)Nx, -0.669183f*(float)Ny, 0.657584f*(float)Nz), -77.0f, 27.0f, 100.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/a/");
			lbm.graphics.set_camera_free(float3(0.224926f*(float)Nx, -0.594332f*(float)Ny, -0.277894f*(float)Nz), -65.0f, -14.0f, 100.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/b/");
			lbm.graphics.set_camera_free(float3(-0.000000f*(float)Nx, 0.650189f*(float)Ny, 1.461048f*(float)Nz), 90.0f, 40.0f, 100.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/c/");
		}
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
	}
} /**/



/*void main_setup() { // Bell 222 helicopter; required extensions in defines.hpp: FP16C, EQUILIBRIUM_BOUNDARIES, MOVING_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 1.2f, 0.3f), 8000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_u = 0.16f;
	const float lbm_length = 0.8f*(float)lbm_N.x;
	const float si_T = 0.34483f; // 2 revolutions of the main rotor
	const ulong lbm_dt = 4ull; // revoxelize rotor every dt time steps
	const float si_length=12.85f, si_d=12.12f, si_rpm=348.0f;
	const float si_u = si_rpm/60.0f*si_d*pif;
	const float si_nu=1.48E-5f, si_rho=1.225f;
	units.set_m_kg_s(lbm_length, lbm_u, 1.0f, si_length, si_u, si_rho);
	const float lbm_nu = units.nu(si_nu);
	const ulong lbm_T = units.t(si_T);
	LBM lbm(lbm_N, 1u, 1u, 1u, lbm_nu);
	// ###################################################################################### define geometry ######################################################################################
	Mesh* body = read_stl(get_exe_path()+"../stl/Bell-222-body.stl"); // https://www.thingiverse.com/thing:1625155/files
	Mesh* main = read_stl(get_exe_path()+"../stl/Bell-222-main.stl"); // body and rotors separated with Microsoft 3D Builder
	Mesh* back = read_stl(get_exe_path()+"../stl/Bell-222-back.stl");
	const float scale = lbm_length/body->get_bounding_box_size().y; // scale body and rotors to simulation box size
	body->scale(scale);
	main->scale(scale);
	back->scale(scale);
	const float3 offset = lbm.center()-body->get_bounding_box_center(); // move body and rotors to simulation box center
	body->translate(offset);
	main->translate(offset);
	back->translate(offset);
	body->set_center(body->get_center_of_mass()); // set rotation center of mesh to its center of mass
	main->set_center(main->get_center_of_mass());
	back->set_center(back->get_center_of_mass());
	const float main_radius=0.5f*main->get_max_size(), main_omega=lbm_u/main_radius, main_domega=main_omega*(float)lbm_dt;
	const float back_radius=0.5f*back->get_max_size(), back_omega=-lbm_u/back_radius, back_domega=back_omega*(float)lbm_dt;
	lbm.voxelize_mesh_on_device(body);
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] =  0.2f*lbm_u;
		if(lbm.flags[n]!=TYPE_S) lbm.u.z[n] = -0.1f*lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_E; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<=lbm_T) { // main simulation loop
		lbm.voxelize_mesh_on_device(main, TYPE_S, main->get_center(), float3(0.0f), float3(0.0f, 0.0f, main_omega)); // revoxelize mesh on GPU
		lbm.voxelize_mesh_on_device(back, TYPE_S, back->get_center(), float3(0.0f), float3(back_omega, 0.0f, 0.0f)); // revoxelize mesh on GPU
		lbm.run(lbm_dt, lbm_T); // run dt time steps
		main->rotate(float3x3(float3(0.0f, 0.0f, 1.0f), main_domega)); // rotate mesh
		back->rotate(float3x3(float3(1.0f, 0.0f, 0.0f), back_domega)); // rotate mesh
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
		if(lbm.graphics.next_frame(lbm_T, 10.0f)) {
			lbm.graphics.set_camera_free(float3(0.528513f*(float)Nx, 0.102095f*(float)Ny, 1.302283f*(float)Nz), 16.0f, 47.0f, 96.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/a/");
			lbm.graphics.set_camera_free(float3(0.0f*(float)Nx, -0.114244f*(float)Ny, 0.543265f*(float)Nz), 90.0f+degrees((float)lbm.get_t()/(float)lbm_dt*main_domega), 36.0f, 120.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/b/");
			lbm.graphics.set_camera_free(float3(0.557719f*(float)Nx, -0.503388f*(float)Ny, -0.591976f*(float)Nz), -43.0f, -21.0f, 75.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/c/");
			lbm.graphics.set_camera_centered(58.0f, 9.0f, 88.0f, 1.648722f);
			lbm.graphics.write_frame(get_exe_path()+"export/d/");
			lbm.graphics.set_camera_centered(0.0f, 90.0f, 100.0f, 1.100000f);
			lbm.graphics.write_frame(get_exe_path()+"export/e/");
			lbm.graphics.set_camera_free(float3(0.001612f*(float)Nx, 0.523852f*(float)Ny, 0.992613f*(float)Nz), 90.0f, 37.0f, 94.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/f/");
		}
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
	}
} /**/



/*void main_setup() { // Mercedes F1 W14 car; required extensions in defines.hpp: FP16S, EQUILIBRIUM_BOUNDARIES, MOVING_BOUNDARIES, SUBGRID, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 2.0f, 0.5f), 4000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_u = 0.075f;
	const float lbm_length = 0.8f*(float)lbm_N.y;
	const float si_T = 0.25f;
	const float si_u = 100.0f/3.6f;
	const float si_length=5.5f, si_width=2.0f;
	const float si_nu=1.48E-5f, si_rho=1.225f;
	units.set_m_kg_s(lbm_length, lbm_u, 1.0f, si_length, si_u, si_rho);
	const float lbm_nu = units.nu(si_nu);
	const ulong lbm_T = units.t(si_T);
	print_info("Re = "+to_string(to_uint(units.si_Re(si_width, si_u, si_nu))));
	LBM lbm(lbm_N, 1u, 1u, 1u, lbm_nu);
	// ###################################################################################### define geometry ######################################################################################
	Mesh* body = read_stl(get_exe_path()+"../stl/mercedesf1-body.stl"); // https://downloadfree3d.com/3d-models/vehicles/sports-car/mercedes-f1-w14/
	Mesh* front_wheels = read_stl(get_exe_path()+"../stl/mercedesf1-front-wheels.stl"); // wheels separated, decals removed and converted to .stl in Microsoft 3D Builder
	Mesh* back_wheels = read_stl(get_exe_path()+"../stl/mercedesf1-back-wheels.stl"); // to avoid instability from too small gaps: remove front wheel fenders and move out right back wheel a bit
	const float scale = lbm_length/body->get_bounding_box_size().y; // scale parts
	body->scale(scale);
	front_wheels->scale(scale);
	back_wheels->scale(scale);
	const float3 offset = float3(lbm.center().x-body->get_bounding_box_center().x, 1.0f-body->pmin.y+0.25f*back_wheels->get_min_size(), 4.0f-back_wheels->pmin.z);
	body->translate(offset);
	front_wheels->translate(offset);
	back_wheels->translate(offset);
	body->set_center(body->get_center_of_mass()); // set rotation center of mesh to its center of mass
	front_wheels->set_center(front_wheels->get_center_of_mass());
	back_wheels->set_center(back_wheels->get_center_of_mass());
	const float lbm_radius=0.5f*back_wheels->get_min_size(), omega=lbm_u/lbm_radius;
	lbm.voxelize_mesh_on_device(body);
	lbm.voxelize_mesh_on_device(front_wheels, TYPE_S, front_wheels->get_center(), float3(0.0f), float3(omega, 0.0f, 0.0f)); // make wheels rotating
	lbm.voxelize_mesh_on_device(back_wheels, TYPE_S, back_wheels->get_center(), float3(0.0f), float3(omega, 0.0f, 0.0f)); // make wheels rotating
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(lbm.flags[n]!=TYPE_S) lbm.u.y[n] = lbm_u;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==Nz-1u) lbm.flags[n] = TYPE_E;
		if(z==0u) lbm.flags[n] = TYPE_S;
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_SURFACE|VIS_Q_CRITERION;
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS)
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<=lbm_T) { // main simulation loop
		if(lbm.graphics.next_frame(lbm_T, 30.0f)) {
			lbm.graphics.set_camera_free(float3(0.779346f*(float)Nx, -0.315650f*(float)Ny, 0.329444f*(float)Nz), -27.0f, 19.0f, 100.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/a/");
			lbm.graphics.set_camera_free(float3(0.556877f*(float)Nx, 0.228191f*(float)Ny, 1.159613f*(float)Nz), 19.0f, 53.0f, 100.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/b/");
			lbm.graphics.set_camera_free(float3(0.220650f*(float)Nx, -0.589529f*(float)Ny, 0.085407f*(float)Nz), -72.0f, 16.0f, 86.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/c/");
			const float progress = (float)lbm.get_t()/(float)lbm_T;
			const float A = 75.0f, B = -160.0f;
			lbm.graphics.set_camera_centered(A+progress*(B-A), -5.0f, 100.0f, 1.648721f);
			lbm.graphics.write_frame(get_exe_path()+"export/d/");
		}
		lbm.run(1u, lbm_T);
	}
#else // GRAPHICS && !INTERACTIVE_GRAPHICS
	lbm.run();
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
} /**/



/*void main_setup() { // hydraulic jump; required extensions in defines.hpp: FP16S, VOLUME_FORCE, EQUILIBRIUM_BOUNDARIES, MOVING_BOUNDARIES, SURFACE, SUBGRID, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint memory = 208u; // GPU VRAM in MB
	const float si_T = 100.0f; // simulated time in [s]

	const float3 si_N = float3(0.96f, 3.52f, 0.96f); // box size in [m]
	const float si_p1 = si_N.y*3.0f/20.0f; // socket length in [m]
	const float si_h1 = si_N.z*2.0f/5.0f; // socket height in [m]
	const float si_h2 = si_N.z*3.0f/5.0f; // water height in [m]

	const float si_Q = 0.25f; // inlet volumetric flow rate in [m^3/s]
	const float si_A_inlet = si_N.x*(si_h2-si_h1); // inlet cross-section area in [m^2]
	const float si_A_outlet = si_N.x*si_h1; // outlet cross-section area in [m^2]
	const float si_u_inlet = si_Q/si_A_inlet; // inlet average flow velocity in [m/s]
	const float si_u_outlet = si_Q/si_A_outlet; // outlet average flow velocity in [m/s]

	float const si_nu = 1.0E-6f; // kinematic shear viscosity [m^2/s]
	const float si_rho = 1000.0f; // water density [kg/m^3]
	const float si_g = 9.81f; // gravitational acceleration [m/s^2]
	//const float si_sigma = 73.81E-3f; // water surface tension [kg/s^2] (no need to use surface tension here)

	const uint3 lbm_N = resolution(si_N, memory); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_u_inlet = 0.075f; // velocity in LBM units for pairing lbm_u with si_u --> lbm_u in LBM units will be equivalent si_u in SI units
	units.set_m_kg_s((float)lbm_N.y, lbm_u_inlet, 1.0f, si_N.y, si_u_inlet, si_rho); // calculate 3 independent conversion factors (m, kg, s)

	const float lbm_nu = units.nu(si_nu); // kinematic shear viscosity
	const ulong lbm_T = units.t(si_T); // how many time steps to compute to cover exactly si_T seconds in real time
	const float lbm_f = units.f(si_rho, si_g); // force per volume
	//const float lbm_sigma = units.sigma(si_sigma); // surface tension (not required here)

	const uint lbm_p1 = to_uint(units.x(si_p1));
	const uint lbm_h1 = to_uint(units.x(si_h1));
	const uint lbm_h2 = to_uint(units.x(si_h2));
	const float lbm_u_outlet = units.u(si_u_outlet);

	LBM lbm(lbm_N, 1u, 1u, 1u, lbm_nu, 0.0f, 0.0f, -lbm_f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(z<lbm_h2) {
			lbm.flags[n] = TYPE_F;
			lbm.rho[n] = units.rho_hydrostatic(0.0005f, z, lbm_h2);
		}
		if(y<lbm_p1&&z<lbm_h1) lbm.flags[n] = TYPE_S;
		if(y<=1u&&x>0u&&x<Nx-1u&&z>=lbm_h1&&z<lbm_h2) {
			lbm.flags[n] = y==0u ? TYPE_S : TYPE_F;
			lbm.u.y[n] = lbm_u_inlet;
		}
		if(y==Ny-1u&&x>0u&&x<Nx-1u&&z>0u) {
			lbm.flags[n] = TYPE_E;
			lbm.u.y[n] = lbm_u_outlet;
		}
		if(x==0u||x==Nx-1u||y==0u||z==0u) lbm.flags[n] = TYPE_S; // sides and bottom non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE;
	lbm.run();
	//lbm.run(1000u); lbm.u.read_from_device(); println(lbm.u.x[lbm.index(Nx/2u, Ny/4u, Nz/4u)]); wait(); // test for binary identity
} /**/



/*void main_setup() { // dam break; required extensions in defines.hpp: FP16S, VOLUME_FORCE, SURFACE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(128u, 256u, 256u, 0.005f, 0.0f, 0.0f, -0.0002f, 0.0001f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(z<Nz*6u/8u && y<Ny/8u) lbm.flags[n] = TYPE_F;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE;
	lbm.run();
} /**/



/*void main_setup() { // liquid metal on a speaker; required extensions in defines.hpp: FP16S, VOLUME_FORCE, MOVING_BOUNDARIES, SURFACE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint L = 128u;
	const float u = 0.09f; // peak velocity of speaker membrane
	const float f = 0.0005f;
	const float frequency = 0.01f; // amplitude = u/(2.0f*pif*frequency);
	LBM lbm(L, L, L*3u/4u, 0.01f, 0.0f, 0.0f, -f, 0.005f);
	// ###################################################################################### define geometry ######################################################################################
	const uint threads = (uint)thread::hardware_concurrency();
	vector<uint> seed(threads);
	for(uint t=0u; t<threads; t++) seed[t] = 42u+t;
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), threads, [&](ulong n, uint t) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(z<Nz/3u && x>0u&&x<Nx-1u&&y>0u&&y<Ny-1u&&z>0u&&z<Nz-1u) {
			lbm.rho[n] = units.rho_hydrostatic(f, (float)z, (float)(Nz/3u));
			lbm.u.x[n] = random_symmetric(seed[t], 1E-9f);
			lbm.u.y[n] = random_symmetric(seed[t], 1E-9f);
			lbm.u.z[n] = random_symmetric(seed[t], 1E-9f);
			lbm.flags[n] = TYPE_F;
		}
		if(z==0u) lbm.u.z[n] = 1E-16f;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE;
	lbm.run(0u); // initialize simulation
	while(true) { // main simulation loop
		lbm.u.read_from_device();
		const float uz = u*sinf(2.0f*pif*frequency*(float)lbm.get_t());
		for(uint z=0u; z<1u; z++) {
			for(uint y=1u; y<Ny-1u; y++) {
				for(uint x=1u; x<Nx-1u; x++) {
					const uint n = x+(y+z*Ny)*Nx;
					lbm.u.z[n] = uz;
				}
			}
		}
		lbm.u.write_to_device();
		lbm.run(1u);
	}
} /**/



/*void main_setup() { // breaking waves on beach; required extensions in defines.hpp: FP16S, VOLUME_FORCE, EQUILIBRIUM_BOUNDARIES, SURFACE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const float f = 0.001f; // make smaller
	const float u = 0.12f; // peak velocity of speaker membrane
	const float frequency = 0.0007f; // amplitude = u/(2.0f*pif*frequency);
	LBM lbm(128u, 640u, 96u, 0.01f, 0.0f, 0.0f, -f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		const uint H = Nz/2u;
		if(z<H) {
			lbm.flags[n] = TYPE_F;
			lbm.rho[n] = units.rho_hydrostatic(f, (float)z, (float)H);
		}
		if(plane(x, y, z, float3(lbm.center().x, 128.0f, 0.0f), float3(0.0f, -1.0f, 8.0f))) lbm.flags[n] = TYPE_S;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // all non periodic
		if(y==0u && x>0u&&x<Nx-1u&&z>0u&&z<Nz-1u) lbm.flags[n] = TYPE_E;
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE | (lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE);
	lbm.run(0u); // initialize simulation
	while(true) { // main simulation loop
		lbm.u.read_from_device();
		const float uy = u*sinf(2.0f*pif*frequency*(float)lbm.get_t());
		const float uz = 0.5f*u*cosf(2.0f*pif*frequency*(float)lbm.get_t());
		for(uint z=1u; z<Nz-1u; z++) {
			for(uint y=0u; y<1u; y++) {
				for(uint x=1u; x<Nx-1u; x++) {
					const uint n = x+(y+z*Ny)*Nx;
					lbm.u.y[n] = uy;
					lbm.u.z[n] = uz;
				}
			}
		}
		lbm.u.write_to_device();
		lbm.run(100u);
	}
} /**/



/*void main_setup() { // river; required extensions in defines.hpp: FP16S, VOLUME_FORCE, SURFACE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(128u, 384u, 96u, 0.02f, 0.0f, -0.00007f, -0.0005f, 0.01f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		const int R = 20, H = 32;
		if(z==0) lbm.flags[n] = TYPE_S;
		else if(z<H) {
			lbm.flags[n] = TYPE_F;
			lbm.u.y[n] = -0.1f;
		}
		if(cylinder(x, y, z, float3(Nx*2u/3u, Ny*2u/3u, Nz/2u)+0.5f, float3(0u, 0u, Nz), (float)R)) lbm.flags[n] = TYPE_S;
		if(cuboid(x, y, z, float3(Nx/3u, Ny/3u, Nz/2u)+0.5f, float3(2u*R, 2u*R, Nz))) lbm.flags[n] = TYPE_S;
		if(x==0u||x==Nx-1u) lbm.flags[n] = TYPE_S; // x non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE;
	lbm.run();
} /**/



/*void main_setup() { // raindrop impact; required extensions in defines.hpp: FP16C, VOLUME_FORCE, EQUILIBRIUM_BOUNDARIES, SURFACE, INTERACTIVE_GRAPHICS or GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(1.0f, 1.0f, 0.85f), 4000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	float lbm_D = (float)lbm_N.x/5.0f;
	const float lbm_u = 0.05f; // impact velocity in LBM units
	const float si_T = 0.003f; // simulated time in [s]
	const float inclination = 20.0f; // impact angle [°], 0 = vertical
	const int select_drop_size = 12;
	//                            0        1        2        3        4        5        6        7        8        9       10       11       12       13 (13 is for validation)
	const float si_Ds[] = { 1.0E-3f, 1.5E-3f, 2.0E-3f, 2.5E-3f, 3.0E-3f, 3.5E-3f, 4.0E-3f, 4.5E-3f, 5.0E-3f, 5.5E-3f, 6.0E-3f, 6.5E-3f, 7.0E-3f, 4.1E-3f };
	const float si_us[] = {   4.50f,   5.80f,   6.80f,   7.55f,   8.10f,   8.45f,   8.80f,   9.05f,   9.20f,   9.30f,   9.40f,   9.45f,   9.55f,   7.21f };
	float const si_nu = 1.0508E-6f; // kinematic shear viscosity [m^2/s] at 20°C and 35g/l salinity
	const float si_rho = 1024.8103f; // fluid density [kg/m^3] at 20°C and 35g/l salinity
	const float si_sigma = 73.81E-3f; // fluid surface tension [kg/s^2] at 20°C and 35g/l salinity
	const float si_g = 9.81f; // gravitational acceleration [m/s^2]
	const float si_D = si_Ds[select_drop_size]; // drop diameter [m] (1-7mm)
	const float si_u = si_us[select_drop_size]; // impact velocity [m/s] (4.50-9.55m/s)
	units.set_m_kg_s(lbm_D, lbm_u, 1.0f, si_D, si_u, si_rho); // calculate 3 independent conversion factors (m, kg, s)
	const float lbm_nu = units.nu(si_nu);
	const ulong lbm_T = units.t(si_T);
	const float lbm_f = units.f(si_rho, si_g);
	const float lbm_sigma = units.sigma(si_sigma);
	print_info("D = "+to_string(si_D, 6u));
	print_info("Re = "+to_string(units.si_Re(si_D, si_u, si_nu), 6u));
	print_info("We = "+to_string(units.si_We(si_D, si_u, si_rho, si_sigma), 6u));
	print_info("Fr = "+to_string(units.si_Fr(si_D, si_u, si_g), 6u));
	print_info("Ca = "+to_string(units.si_Ca(si_u, si_rho, si_nu, si_sigma), 6u));
	print_info("Bo = "+to_string(units.si_Bo(si_D, si_rho, si_g, si_sigma), 6u));
	print_info(to_string(to_uint(1000.0f*si_T))+" ms = "+to_string(units.t(si_T))+" LBM time steps");
	const float lbm_H = 0.4f*(float)lbm_N.x;
	const float lbm_R = 0.5f*lbm_D; // drop radius
	LBM lbm(lbm_N, 1u, 1u, 1u, lbm_nu, 0.0f, 0.0f, -lbm_f, lbm_sigma); // calculate values for remaining parameters in simulation units
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(sphere(x, y, z, float3(0.5f*(float)Nx, 0.5f*(float)Ny-2.0f*lbm_R*tan(inclination*pif/180.0f), lbm_H+lbm_R+2.5f)+0.5f, lbm_R+2.0f)) {
			const float b = sphere_plic(x, y, z, float3(0.5f*(float)Nx, 0.5f*(float)Ny-2.0f*lbm_R*tan(inclination*pif/180.0f)+0.5f, lbm_H+lbm_R+2.5f), lbm_R);
			if(b!=-1.0f) {
				lbm.u.y[n] =  sinf(inclination*pif/180.0f)*lbm_u;
				lbm.u.z[n] = -cosf(inclination*pif/180.0f)*lbm_u;
				if(b==1.0f) {
					lbm.flags[n] = TYPE_F;
					lbm.phi[n] = 1.0f;
				} else {
					lbm.flags[n] = TYPE_I;
					lbm.phi[n] = b; // initialize cell fill level phi directly instead of just flags, this way the raindrop sphere is smooth already at initialization
				}
			}
		}
		if(z==0) lbm.flags[n] = TYPE_S;
		else if(z==to_uint(lbm_H)) {
			lbm.flags[n] = TYPE_I;
			lbm.phi[n] = 0.5f; // not strictly necessary, but should be clearer (phi is automatically initialized to 0.5f for TYPE_I if not initialized)
		} else if((float)z<lbm_H) lbm.flags[n] = TYPE_F;
		else if((x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==Nz-1u)&&(float)z>lbm_H+0.5f*lbm_R) { // make drops that hit the simulation box ceiling disappear
			lbm.rho[n] = 0.5f;
			lbm.flags[n] = TYPE_E;
		}
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE;
#if defined(GRAPHICS) && !defined(INTERACTIVE_GRAPHICS) && !defined(INTERACTIVE_GRAPHICS_ASCII)
	lbm.run(0u, lbm_T); // initialize simulation
	while(lbm.get_t()<=lbm_T) { // main simulation loop
		if(lbm.graphics.next_frame(lbm_T, 20.0f)) { // generate video
			lbm.graphics.set_camera_centered(-30.0f, 20.0f, 100.0f, 1.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/n/");
			lbm.graphics.set_camera_centered(10.0f, 40.0f, 100.0f, 1.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/p/");
			lbm.graphics.set_camera_centered(0.0f, 0.0f, 45.0f, 1.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/o/");
			lbm.graphics.set_camera_centered(0.0f, 90.0f, 45.0f, 1.0f);
			lbm.graphics.write_frame(get_exe_path()+"export/t/");
		}
		lbm.run(1u, lbm_T);
	}
	//lbm.run(lbm_T); // only generate one image
	//lbm.graphics.set_camera_centered(-30.0f, 20.0f, 100.0f, 1.0f);
	//lbm.graphics.write_frame();
#else // GRAPHICS && !INTERACTIVE_GRAPHICS
	lbm.run();
#endif // GRAPHICS && !INTERACTIVE_GRAPHICS
} /**/



/*void main_setup() { // bursting bubble; required extensions in defines.hpp: FP16C, VOLUME_FORCE, SURFACE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	const uint3 lbm_N = resolution(float3(4.0f, 4.0f, 3.0f), 1000u); // input: simulation box aspect ratio and VRAM occupation in MB, output: grid resolution
	const float lbm_d = 0.25f*(float)lbm_N.x; // bubble diameter in LBM units
	const float lbm_sigma = 0.0003f; // surface tension coefficient in LBM units
	const float si_nu = 1E-6f; // kinematic shear viscosity (water) [m^2/s]
	const float si_rho = 1E3f; // density (water) [kg/m^3]
	const float si_sigma = 0.072f; // surface tension (water) [kg/s^2]
	const float si_d = 4E-3f; // bubble diameter [m]
	const float si_g = 9.81f; // gravitational acceleration [m/s^2]
	const float si_f = units.si_f_from_si_g(si_g, si_rho);
	const float lbm_rho = 1.0f;
	const float m = si_d/lbm_d; // length si_x = x*[m]
	const float kg = si_rho/lbm_rho*cb(m); // density si_rho = rho*[kg/m^3]
	const float s = sqrt(lbm_sigma/si_sigma*kg); // velocity si_sigma = sigma*[kg/s^2]
	units.set_m_kg_s(m, kg, s); // do unit conversion manually via d, rho and sigma
	const uint lbm_H = to_uint(2.0f*lbm_d);
	LBM lbm(lbm_N, units.nu(si_nu), 0.0f, 0.0f, -units.f(si_f), lbm_sigma);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(z<lbm_H) lbm.flags[n] = TYPE_F;
		const float r = 0.5f*lbm_d;
		if(sphere(x, y, z, float3(lbm.center().x, lbm.center().y, (float)lbm_H-0.5f*lbm_d), r+1.0f)) { // bubble
			const float b = clamp(sphere_plic(x, y, z, float3(lbm.center().x, lbm.center().y, (float)lbm_H-0.5f*lbm_d), r), 0.0f, 1.0f);
			if(b==1.0f) {
				lbm.flags[n] = TYPE_G;
				lbm.phi[n] = 0.0f;
			} else {
				lbm.flags[n] = TYPE_I;
				lbm.phi[n] = (1.0f-b); // initialize cell fill level phi directly instead of just flags, this way the bubble sphere is smooth already at initialization
			}
		}
		if(z==0) lbm.flags[n] = TYPE_S;
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE;
	lbm.run();
} /**/



/*void main_setup() { // cube with changing gravity; required extensions in defines.hpp: FP16S, VOLUME_FORCE, SURFACE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(96u, 96u, 96u, 0.02f, 0.0f, 0.0f, -0.001f, 0.001f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(x<Nx*2u/3u&&y<Ny*2u/3u) lbm.flags[n] = TYPE_F;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S;
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE;
	lbm.run(0u); // initialize simulation
	while(true) { // main simulation loop
		lbm.set_f(0.0f, 0.0f, -0.001f);
		lbm.run(2500u);
		lbm.set_f(0.0f, +0.001f, 0.0f);
		lbm.run(2500u);
		lbm.set_f(0.0f, 0.0f, +0.001f);
		lbm.run(2500u);
		lbm.set_f(0.0f, -0.001f, 0.0f);
		lbm.run(2000u);
		lbm.set_f(0.0f, 0.0f, 0.0f);
		lbm.run(3000u);
	}
} /**/



/*void main_setup() { // periodic faucet mass conservation test; required extensions in defines.hpp: FP16S, VOLUME_FORCE, SURFACE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(96u, 192u, 128u, 0.02f, 0.0f, 0.0f, -0.00025f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(y>Ny*5u/6u) lbm.flags[n] = TYPE_F;
		const uint D=max(Nx, Nz), R=D/6;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u) lbm.flags[n] = TYPE_S; // x and y non periodic
		if((z==0u||z==Nz-1u) && sq(x-Nx/2)+sq(y-Nx/2)>sq(R)) lbm.flags[n] = TYPE_S; // z non periodic
		if(y<=Nx/2u+2u*R && torus_x(x, y, z, float3(Nx/2u, Nx/2u+R, Nz)+0.5f, (float)R, (float)R)) lbm.flags[n] = TYPE_S;
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_PHI_RASTERIZE;
	lbm.run();
} /**/



/*void main_setup() { // two colliding droplets in force field; required extensions in defines.hpp: FP16S, VOLUME_FORCE, FORCE_FIELD, SURFACE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(256u, 256u, 128u, 0.014f, 0.0f, 0.0f, 0.0f, 0.0001f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(sphere(x, y, z, lbm.center()-float3(0u, 10u, 0u), 32.0f)) {
			lbm.flags[n] = TYPE_F;
			lbm.u.y[n] = 0.025f;
		}
		if(sphere(x, y, z, lbm.center()+float3(30u, 40u, 0u), 12.0f)) {
			lbm.flags[n] = TYPE_F;
			lbm.u.y[n] = -0.2f;
		}
		lbm.F.x[n] = -0.001f*lbm.relative_position(n).x;
		lbm.F.y[n] = -0.001f*lbm.relative_position(n).y;
		lbm.F.z[n] = -0.0005f*lbm.relative_position(n).z;
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = lbm.get_D()==1u ? VIS_PHI_RAYTRACE : VIS_PHI_RASTERIZE;
	lbm.run();
} /**/



/*void main_setup() { // Rayleigh-Benard convection; required extensions in defines.hpp: FP16S, VOLUME_FORCE, TEMPERATURE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(256u, 256u, 64u, 1u, 1u, 1u, 0.02f, 0.0f, 0.0f, -0.0005f, 0.0f, 1.0f, 1.0f);
	// ###################################################################################### define geometry ######################################################################################
	const uint threads = (uint)thread::hardware_concurrency();
	vector<uint> seed(threads);
	for(uint t=0u; t<threads; t++) seed[t] = 42u+t;
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), threads, [&](ulong n, uint t) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		lbm.u.x[n] = random_symmetric(seed[t], 0.015f); // initialize velocity with random noise
		lbm.u.y[n] = random_symmetric(seed[t], 0.015f);
		lbm.u.z[n] = random_symmetric(seed[t], 0.015f);
		lbm.rho[n] = units.rho_hydrostatic(0.0005f, (float)z, 0.5f*(float)Nz); // initialize density with hydrostatic pressure
		if(z==1u) {
			lbm.T[n] = 1.75f;
			lbm.flags[n] = TYPE_T;
		} else if(z==Nz-2u) {
			lbm.T[n] = 0.25f;
			lbm.flags[n] = TYPE_T;
		}
		if(z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // leave lateral simulation box walls periodic by not closing them with TYPE_S
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_STREAMLINES;
	lbm.run();
} /**/



/*void main_setup() { // thermal convection; required extensions in defines.hpp: FP16S, VOLUME_FORCE, TEMPERATURE, INTERACTIVE_GRAPHICS
	// ################################################################## define simulation box size, viscosity and volume force ###################################################################
	LBM lbm(32u, 196u, 60u, 1u, 1u, 1u, 0.02f, 0.0f, 0.0f, -0.0005f, 0.0f, 1.0f, 1.0f);
	// ###################################################################################### define geometry ######################################################################################
	const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
		if(y==1) {
			lbm.T[n] = 1.8f;
			lbm.flags[n] = TYPE_T;
		} else if(y==Ny-2) {
			lbm.T[n] = 0.3f;
			lbm.flags[n] = TYPE_T;
		}
		lbm.rho[n] = units.rho_hydrostatic(0.0005f, (float)z, 0.5f*(float)Nz); // initialize density with hydrostatic pressure
		if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==0u||z==Nz-1u) lbm.flags[n] = TYPE_S; // all non periodic
	}); // ####################################################################### run simulation, export images and data ##########################################################################
	lbm.graphics.visualization_modes = VIS_FLAG_LATTICE|VIS_STREAMLINES;
	lbm.run();
	//lbm.run(1000u); lbm.u.read_from_device(); println(lbm.u.x[lbm.index(Nx/2u, Ny/2u, Nz/2u)]); wait(); // test for binary identity
} /**/