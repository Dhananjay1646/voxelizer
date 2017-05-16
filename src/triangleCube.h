#ifndef TRIANGLECUBE_H
#define TRIANGLECUBE_H

#include <Eigen/Dense>

struct Point3{
	Point3(Eigen::Vector3f v)
	{
		x = v(0); y = v(1); z = v(2);
	}

	Point3(float v0, float v1, float v2)
	{
		x = v0; y = v1; z = v2;
	}

	Point3()
	{
		x = 0; y = 0; z = 0;
	}
	 
	  
	float x;
	float y;
	float z;
};

inline Eigen::Vector3f getVector3f(const Point3 p)
{
	return Eigen::Vector3f(p.x, p.y, p.z);
}

struct Triangle3{
	Point3 v1;                 /* Vertex1 */
	Point3 v2;                 /* Vertex2 */
	Point3 v3;                 /* Vertex3 */

	Triangle3 translate(Eigen::Vector3f v)
	{
	   Eigen::Vector3f ev1 = getVector3f(v1);
	   Eigen::Vector3f ev2 = getVector3f(v2);
	   Eigen::Vector3f ev3 = getVector3f(v3);

	   ev1 += v;
	   ev2 += v;
	   ev3 += v;

	   Point3 new_v1 = Point3(ev1);
	   Point3 new_v2 = Point3(ev2);
	   Point3 new_v3 = Point3(ev3);

	   Triangle3 result = {new_v1, new_v2, new_v3};
	   return result;
	}

	Triangle3 scale(float s)
	{
	   Eigen::Vector3f ev1 = getVector3f(v1);
	   Eigen::Vector3f ev2 = getVector3f(v2);
	   Eigen::Vector3f ev3 = getVector3f(v3);

	   ev1 *= s;
	   ev2 *= s;
	   ev3 *= s;

	   Point3 new_v1 = Point3(ev1);
	   Point3 new_v2 = Point3(ev2);
	   Point3 new_v3 = Point3(ev3);

	   Triangle3 result = {new_v1, new_v2, new_v3};
	   return result;
	}
}; 
 

long t_c_intersection(Triangle3 t);

#endif
