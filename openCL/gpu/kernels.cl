__kernel void make_voxel_borders(
	__global float *borders, float lower_lim, float upper_lim, int steps)
{
			int ix = get_global_id(0);
			float range = upper_lim - lower_lim;
			float frac = 1.0f - (float)(steps-1-ix)/ (float)(steps-1);
			borders[ix] = lower_lim + range*frac;
}


__kernel void make_uniform_circular_distributed_points(
	__global float4 *r0, __global float *h, __global float *w, uint seedx,
	uint seedy, float radius)
{
			int ix = get_global_id(0);
			int maxv = ((2<<31)-1);
			int counter = 0;
			float xpos, ypos;
			uint x = seedx + ix + 1;
			uint y = seedy + ix + 1;
			uint resx, resy;
			do {
				x ^= x >> 12;
				x ^= x << 25;
				x ^= x >> 27;
				resx = x*0x2545F4914F6CDD1D;
				y ^= y >> 12;
				y ^= y << 25;
				y ^= y >> 27;
				resy = y*0x2545F4914F6CDD1D;
				xpos = (2*(resx / 4294967295.0f)-1) * radius;
				ypos = (2*(resy / 4294967295.0f)-1) * radius;
				//counter += 1;
			}
			while(length((float2)(xpos, ypos)) > radius);
			//while(counter < 10);
			h[ix] =  xpos;
			w[ix] =  ypos;
			r0[ix] = 1.0f;
}

__kernel void make_circular_gaussian_distributed_points(
__global float4 *r0,	__global float *h, __global float *w, uint seedx, uint seedy, float radius)
{
			int ix = get_global_id(0);
			int maxv = ((2<<31)-1);
			int counter = 0;
			float xpos, ypos, rad;
      uint x = seedx + ix;
			uint y = seedy + ix;
			uint resx, resy;
			x ^= x >> 12;
			x ^= x << 25;
			x ^= x >> 27;
			resx = x*0x2545F4914F6CDD1D;
			y ^= y >> 12;
			y ^= y << 25;
			y ^= y >> 27;
			resy = y*0x2545F4914F6CDD1D;
			float a = sqrt(-2*log(resx / 4294967295.0f));
			xpos = a * cos(resy/ 4294967295.0f*2*M_PI) * radius;
			ypos = a * sin(resy/ 4294967295.0f*2*M_PI) * radius;

      // float l = length(newpos);
      // rad = length(newpos);
      // rad = 0.5f*(1+erf(rad/sigma))*rad;
      // newpos = rad*newpos / length(newpos);

			//while(counter < 10);
			h[ix] =  xpos;
			w[ix] =  ypos;
			r0[ix] = 1.0f;
}


__kernel void make_square_gaussian_distributed_points(
	__global float4 *r0, __global float *h, __global float *w, uint seedx, uint seedy, float radius,
	float width, float height)
{
			int ix = get_global_id(0);
			int maxv = ((2<<31)-1);
			int counter = 0;
			int max_iterations = 2500;
			float xpos, ypos, rad;
      uint x = seedx + ix; // Add +1, otherwise kernel will crash for ix = 0
			uint y = seedy + ix; // Add +1, otherwise kernel will crash for ix = 0
			uint resx, resy;
			while(counter < max_iterations)
			{
				x ^= x >> 12;
				x ^= x << 25;
				x ^= x >> 27;
				resx = x*0x2545F4914F6CDD1D;
				y ^= y >> 12;
				y ^= y << 25;
				y ^= y >> 27;
				resy = y*0x2545F4914F6CDD1D;
				float b = resx / 4294967295.0f;
				float a = sqrt(-2*log(resx / 4294967295.0f));
				xpos = a * cos(resy/ 4294967295.0f*2.0f*M_PI) * radius;
				ypos = a * sin(resy/ 4294967295.0f*2.0f*M_PI) * radius;

				counter += 1;
				if (fabs(ypos) < (width / 2.0f) && fabs(xpos) < (height / 2.0f)){
					break;
				}
			}

			if (counter == max_iterations){
					r0[ix].w = -1.0f;
			}
			else {
					r0[ix].w = 1.0f;
			}
			h[ix] =  xpos;
			w[ix] =  ypos;

}


bool intersect_cylinder(
	__global float4 *p1, __global float4 *p2, __global float4 *intersection_point,
	float radius, bool nearest
	){
	float2 d = {p2->z-p1->z, p2->y-p1->y};
	float dr = length(d);
	float det = p1->z*p2->y-p2->z*p1->y;
	float discriminant = pown(radius,2)*pown(dr,2) - pown(det,2);

	if(discriminant < 0){
		// Imaginary solution
		return false;
	}
	else
	{
		// 1 or 2 solutions
		float root = sqrt(discriminant);
		float rsqr = pown(dr,2);
		float4 v = *p2-*p1;
		float4 inter0 = {0,(-det*d.x+fabs(d.y)*root)/rsqr, (det*d.y+sign(d.y)*d.x*root)/rsqr,0};
		float t0 = (inter0.y-p1->y)/(v.y);
		inter0.x = p1->x + t0*v.x;
		float4 inter1 = {0,(-det*d.x-fabs(d.y)*root)/rsqr, (det*d.y-sign(d.y)*d.x*root)/rsqr,0};
		float t1 = (inter1.y-p1->y)/(v.y);
		inter1.x = p1->x + t1*v.x;

		// choose nearest intersection point
		if(nearest){
			if (distance(inter0.xyz, p1->xyz) <= distance(inter1.xyz, p1->xyz)){
				intersection_point->xyz = inter0.xyz;
			}
			else
			{
				intersection_point->xyz = inter1.xyz;
			}
		}
		else
		{
			if (distance(inter0.xyz, p1->xyz) <= distance(inter1.xyz, p1->xyz)){
				intersection_point->xyz = inter1.xyz;
			}
			else
			{
				intersection_point->xyz = inter0.xyz;
			}
		}

		return true;
	}
}


bool intersect_rectangle(
	__global float4 *p1, __global float4 *p2, __global float4 *intersection_point,
	float thickness, float width, float angle, bool nearest
	){
		float2 n = {-1.0f*cos(angle), sin(angle)};
		float d = thickness / 2.0f;
		float4 v = *p2-*p1;
		// choose nearest intersection point
		// if(nearest){
		// 	if (distance(inter0.xyz, p1->xyz) <= distance(inter1.xyz, p1->xyz)){
		// 		intersection_point->xyz = inter0.xyz;
		// 	}
		// 	}
		// 	else
		// 	{
		// 		intersection_point->xyz = inter1.xyz;
		// 	}
		// }
		// else
		// {
		// 	if (distance(inter0.xyz, p1->xyz) <= distance(inter1.xyz, p1->xyz)){
		// 		intersection_point->xyz = inter1.xyz;
		// 	}
		// 	else
		// 	{
		// 		intersection_point->xyz = inter0.xyz;
		// 	}
		// }

		return true;
}


float4 fresnel(
	float4 normal, float4 vec, float n1, float n2, float *transmitted
	){
	float angle = acos(dot(vec.xyz, normal.xyz) / (length(normal.xyz)*length(vec.xyz)));
	float theta_i = M_PI - angle;
	float theta_tr= asin(n1/n2 * sin(theta_i));
	float r_s = fabs(n1*cos(theta_i)-n2*cos(theta_tr)/(n1*cos(theta_i)+n2*cos(theta_tr)));
	float r_p = fabs(n1*cos(theta_tr)-n2*cos(theta_i)/(n1*cos(theta_tr)+n2*cos(theta_i)));
	*transmitted = 1.0f-(pown(r_s,2)+pown(r_p,2))/2.0f;


	float4 axis = cross(normal, vec);
	float length = sqrt(pown(axis.x,2)+pown(axis.y,2)+pown(axis.z,2));
	float4 u = {axis.x/length, axis.y/length, axis.z/length, axis.w/length};


	// === compute transmitted vector ===
	float sin_th_r = n1/n2 * sin(theta_i);
	float sin_th_i = sin(M_PI - angle);
	float cos_th_r = sqrt(1-pown(sin_th_r, 2));
	float cos_th_i = cos(M_PI - angle);
	float sin_th_diff = sin_th_i*cos_th_r-cos_th_i*sin_th_r;
	float cos_th_diff = cos_th_i*cos_th_r+sin_th_i*sin_th_r;
	// Build rotation matrix

	float4 T1;
	float4 T2;
	float4 T3;
	T1.x = cos_th_diff+pown(u.x,2)*(1-cos_th_diff);
	T1.y = u.x*u.y*(1-cos_th_diff)-u.z*sin_th_diff;
	T1.z = u.x*u.z*(1-cos_th_diff)+u.y*sin_th_diff;
	T2.x = u.y*u.x*(1-cos_th_diff)+u.z*sin_th_diff;
	T2.y = cos_th_diff+pown(u.y,2)*(1-cos_th_diff);
	T2.z = u.y*u.z*(1-cos_th_diff)-u.x*sin_th_diff;
	T3.x = u.z*u.x*(1-cos_th_diff)-u.y*sin_th_diff;
	T3.y = u.z*u.y*(1-cos_th_diff)+u.x*sin_th_diff;
	T3.z = cos_th_diff+pown(u.z,2)*(1-cos_th_diff);

	float4 rotated = {0.0f,0.0f,0.0f,-1.0f};
	rotated.x = T1.x*vec.x+T1.y*vec.y+T1.z*vec.z;
	rotated.y = T2.x*vec.x+T2.y*vec.y+T2.z*vec.z;
	rotated.z = T3.x*vec.x+T3.y*vec.y+T3.z*vec.z;

	return rotated;
}


__kernel void path_trace_round_jet(
	__global float4 *r0, __global float4 *r1, __global float4 *r2, __global float4 *focal_point,
	float jet_radius, float4 origin, __global float *h, __global float *w, float n1,
	float n2,	float photons_per_ray
	){

		int ix = get_global_id(0);
		if(r0[ix].w > 0.0f){
			float4 normal = -1.0f*origin;
			float4 u2 = {1,0,0,0};
			float4 u3 = cross(normal, u2);
			float4 v1, v2, v3;
			v1 = normal;
			v2 = cross(normal, u2);
			v3 = cross(v1, v2);
			v1 /= length(v1);
			v2 /= length(v2);
			v3 /= length(v3);
			r0[ix].xyz = origin.xyz + w[ix]*v2.xyz +  h[ix]*v3.xyz;

			// Check if intersection exists
			if(intersect_cylinder(&r0[ix], &focal_point[0], &r1[ix], jet_radius, true))
			{
				float4 normal = {0, r1[ix].s12,0};
				float4 v = r1[ix] - r0[ix];
				float transmitted;
				r2[ix].xyz = fresnel(normal, v, n1, n2, &transmitted).xyz;
				if(intersect_cylinder(&r1[ix], &r2[ix], &r2[ix], jet_radius, false))
				{
					r2[ix].w = transmitted*photons_per_ray;
				}
				else
				{
					r2[ix].w = -1.0f;
				}
			}
			else
			{
				r1[ix].w = -1.0f;
				r2[ix].w = -1.0f;
			}
		}
	}

	__kernel void path_trace_rect_jet(
		__global float4 *r0, __global float4 *r1, __global float4 *r2, __global float4 *focal_point,
		float jet_thickness, float jet_width, float jet_angle, float4 origin, __global float *h,
		__global float *w, float n1, float n2,	float photons_per_ray
		){
			int ix = get_global_id(0);
			if(r0[ix].w > 0.0f) {

			}


			float4 normal = -1.0f*origin;
			float4 u2 = {1,0,0,0};
			float4 u3 = cross(normal, u2);
			float4 v1, v2, v3;
			v1 = normal;
			v2 = cross(normal, u2);
			v3 = cross(v1, v2);
			v1 /= length(v1);
			v2 /= length(v2);
			v3 /= length(v3);
			r0[ix].xyz = origin.xyz + w[ix]*v2.xyz +  h[ix]*v3.xyz;
			//
			// // Check if intersection exists
			// if(intersect_cylinder(&r0[ix], &focal_point[0], &r1[ix], jet_radius, true))
			// {
			// 	float4 normal = {0, r1[ix].s12,0};
			// 	float4 v = r1[ix] - r0[ix];
			// 	float transmitted;
			// 	r2[ix].xyz = fresnel(normal, v, n1, n2, &transmitted).xyz;
			// 	if(intersect_cylinder(&r1[ix], &r2[ix], &r2[ix], jet_radius, false))
			// 	{
			// 		r2[ix].w = transmitted*photons_per_ray;
			// 	}
			// 	else
			// 	{
			// 		r2[ix].w = -1.0f;
			// 	}
			// }
			// else
			// {
			// 	r1[ix].w = -1.0f;
			// 	r2[ix].w = -1.0f;
			// }
		}

	// __kernel void get_r2(
	// 		__global float4 *r0, __global float4 *r1, __global float4 *r2, float jet_radius,
	// 		float n1, float n2,	float photons_per_ray
	// ){
	// 		/////// Jet origin always at (0,0,0,0) ///////
	// 		int ix = get_global_id(0);
	// 		float4 intersection_point;
	// 		if(r1[ix].w > 0){
	// 			float transmitted = -1.0f;
	// 			float4 normal = {0, r1[ix].s12,0};
	// 			float4 v = r1[ix] - r0[ix];
	// 			r2[ix] = fresnel(normal, v, n1, n2, &transmitted);
	// 			if(intersect_cylinder(&r1[ix], &r2[ix], &r2[ix], jet_radius, false))
	// 			{
	// 				r2[ix].w = transmitted*photons_per_ray;
	// 			}
	// 			else
	// 			{
	// 				r2[ix].w = -1.0f;
	// 			}
	// 		}
	// 	}

// __kernel void intersect_r1(
// 	__global float8 *r1, float jet_radius, float n1, float n2, float photons_per_ray
// 	){
// 		/////// Jet origin always at (0,0,0,0) ///////
// 		int ix = get_global_id(0);
// 		float4 intersection_point;
//
// 		// Check if intersection exists
// 		if(intersect_cylinder(r1[ix].lo, r1[ix].hi, &intersection_point, jet_radius, true)){
// 			r2[ix].lo = intersection_point;
// 			float transmitted = -1.0f;
// 			float4 normal = {0, r2[ix].s12,0};
// 			float4 v = r1[ix].hi - r1[ix].lo;
// 			float4 transmitted_ray = fresnel(normal, v, n1, n2, &transmitted);
// 			intersect_cylinder(r2[ix].lo, transmitted_ray, &intersection_point, jet_radius, false);
// 			r2[ix].hi = transmitted_ray;
// 			r2[ix].s7 = transmitted*photons_per_ray;
// 		}
// 		else
// 		{
// 			r2[ix].s7 = -1.0f;
// 		}
// 	}

int get_lin_id(int x, int y, int z, int voxels_per_side){
	if (x >= 0 && x < voxels_per_side &&
			y >= 0 && y < voxels_per_side &&
			z >= 0 && z < voxels_per_side)
	{
		return x + voxels_per_side*y + voxels_per_side*voxels_per_side*z;
	}
	else
	{
		return -1;
	}
}

	__kernel void make_voxels(
		__global float8 *voxels, __global int8 *sides, int voxels_per_side,
		__global float *borders){

		int ix = get_global_id(0);

		// set cube borders
		int x = ix % voxels_per_side;
		int y = (ix-x) / voxels_per_side % voxels_per_side;
		int z = (ix -x-voxels_per_side*y) / (voxels_per_side*voxels_per_side);
		float x_min = borders[x];
		float x_max = borders[x+1];
		float y_min = borders[y];
		float y_max = borders[y+1];
		float z_min = borders[z];
		float z_max = borders[z+1];

		voxels[ix] = (float8) (x_min, x_max, y_min, y_max, z_min, z_max, 0,0);


		// set cube neighbor ids
		sides[ix].s0 = ix;
		sides[ix].s1 = get_lin_id(x,y-1,z, voxels_per_side); // front
		sides[ix].s2 = get_lin_id(x,y+1,z, voxels_per_side); // back
		sides[ix].s3 = get_lin_id(x,y,z+1, voxels_per_side); // top
		sides[ix].s4 = get_lin_id(x,y,z-1, voxels_per_side); // bottom
		sides[ix].s5 = get_lin_id(x-1,y,z, voxels_per_side); // left
		sides[ix].s6 = get_lin_id(x+1,y,z, voxels_per_side); // right
}

int get_border_index(__global float *borders, float position, int voxels_per_side){
	for(int i = 0; i < voxels_per_side; i++){
		if(position >= borders[i] && position <= borders[i+1]){
			return i;
		}
	}
	return -1;
}

// http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html
// The above version is supposedly not always safe: Consider using a different version from
// https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/

inline void AtomicAdd(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void get_photons_per_ray(__global float4 *r0, __global float *counter){
	int ix = get_global_id(0);
	if(r0[ix].w > 0.0f){
		AtomicAdd(&counter[0], 1.0f);
	}
}

__kernel void trace_ray_through_voxels(
	__global float4 *r1, __global float4 *r2, __global float8 *voxels,
	__global int8 *sides, int voxels_per_side, __global float *borders,
	__global float *absorbed_photons, __global float *transmitted_photons,
	float sample_absorption, float solvent_absorption
){
		int ix = get_global_id(0);
		// absorbed_photons[ix] = 0.0f; << EVIL!!!
		// transmitted_photons[ix] = 0.0f; << EVIL!!!
		uint trace_limit = voxels_per_side * 100;

		if(r2[ix].w > 0.0f){
			// only do voxel tracing if there is some ray intensity

			int x_index = get_border_index(borders, r1[ix].x, voxels_per_side);
			int y_index = get_border_index(borders, r1[ix].y, voxels_per_side);
			int z_index = get_border_index(borders, r1[ix].z, voxels_per_side);
			int x_index_s = get_border_index(borders, r2[ix].x, voxels_per_side);
			int y_index_s = get_border_index(borders, r2[ix].y, voxels_per_side);
			int z_index_s = get_border_index(borders, r2[ix].z, voxels_per_side);
			int voxid = get_lin_id(y_index, z_index, x_index, voxels_per_side);
			int voxid_stop = get_lin_id(y_index_s, z_index_s, x_index_s, voxels_per_side);
			int new_voxid;

			// checker[ix] = x_index;

			float tx;
			float ty;
			float tz;
			float4 p0 = (float4)(r1[ix].xyz,0);
			float4 p1 = (float4)(r2[ix].xyz,0);
			float vx = r2[ix].x - r1[ix].x;
			float vy = r2[ix].y - r1[ix].y;
			float vz = r2[ix].z - r1[ix].z;
			// int counter = 0;
			float t;
			float i0 = r2[ix].w; // Transmittance at voxel entrance, changed during loop
			float i1 = 1.0f; // Transmittance at voxel exit, changed during loop

			float jet_extinction = 0.1f*(sample_absorption+solvent_absorption);
			uint counter = 0;

			while(voxid != -1 && voxid != voxid_stop && counter < trace_limit){

				float voxx, voxy, voxz;
				if (vx < 0) voxx = voxels[voxid].s4;
				else voxx = voxels[voxid].s5;
				if (vy < 0) voxy = voxels[voxid].s0;
				else voxy = voxels[voxid].s1;
				if (vz < 0) voxz = voxels[voxid].s2;
				else voxz = voxels[voxid].s3;
				tx = (voxx - p0.x)/vx;
				ty = (voxy - p0.y)/vy;
				tz = (voxz - p0.z)/vz;

				float sidex, sidey, sidez;
				if (vx < 0) sidex = sides[voxid].s4;
				else sidex = sides[voxid].s3;
				if (vy < 0) sidey = sides[voxid].s5;
				else sidey = sides[voxid].s6;
				if (vz < 0) sidez = sides[voxid].s1;
				else sidez = sides[voxid].s2;
				if(tx < ty && tx < tz){
					t = tx;
					new_voxid = sidex;
				}
				else if(ty < tx && ty < tz){
					t = ty;
					new_voxid = sidey;
				}
				else if(tz < tx && tz < ty){
					t = tz;
					new_voxid = sidez;
				}
				p1 = p0 + t*(float4)(vx,vy,vz,0);
				float l = distance(p1,p0);
				p0 = p1;
				i1 = i0*exp10(-l*jet_extinction);
				float i1_sample = i0*exp10(-l*0.1f*sample_absorption);
				// float transmittance = exp10(-l*jet_extinction);
				// float od = l * 0.1 * attcoeff_sample * sample_concentration;
				// float solvent_transmittance = transmittance / exp10(-l*attcoeff_solvent*solvent_concentration);
				// float abs_sample = i0-i1/(i0*exp(-l*attcoeff_solvent*solvent_concentration));
				// atomic_add(&absorbed_photons[voxid], (int)(i0-i1));
				AtomicAdd(&absorbed_photons[voxid], i0-i1_sample);
				AtomicAdd(&transmitted_photons[voxid], i0);
				// atomic_xchg(&intensities[voxid],abs_sample);
				i0 = i1;
				voxid = new_voxid;
				counter += 1;
			}
	 }
}
