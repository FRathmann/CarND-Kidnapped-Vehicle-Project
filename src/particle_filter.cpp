/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <vector>

#include "particle_filter.h"
#include "helper_functions.h"

#define EPS 0.00001

using std::vector;
using std::string;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Number of total particles
  num_particles = 100;
    
  // Resize weights vector based on number of particles
  weights.resize(num_particles);
    
  // Resize vector of particles
  particles.resize(num_particles);
  
  // Engine for later generation of particles
  random_device rd;
  default_random_engine gen(rd());
    
  // Creates a normal distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
    
  // Initializes particles → from the normal distributions
  for (int i = 0; i < num_particles; ++i) 
  {    
    // Add generated particle data to particles class
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }
    
  // Show as initialized; no need for prediction yet
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // Engine for later generation of particles
  default_random_engine gen;
  
  // Make distributions for adding noises
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  // Different equations based on yaw_rate
  for (int i = 0; i < num_particles; ++i) 
  { 
    if (abs(yaw_rate) != 0) 
    {
      // Add measurements to particles
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
    } 
    else 
    {
      // Add measurements to particles
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }

    // Add noise to the particles
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  unsigned int nObservations = observations.size();
  unsigned int nPredictions = predicted.size();

  for (unsigned int i = 0; i < nObservations; i++) 
  { 
    // Initialize min distance → make it a big number
    double minDistance = numeric_limits<double>::max();

    // Initialize the found map with a default value
    int mapId = -1;

    for (unsigned j = 0; j < nPredictions; j++ ) 
    {
      double xDistance = observations[i].x - predicted[j].x;
      double yDistance = observations[i].y - predicted[j].y;

      double distance = xDistance * xDistance + yDistance * yDistance;

      // Distance is less than min → stored the id and update min
      if ( distance < minDistance ) 
      {
        minDistance = distance;
        mapId = predicted[j].id;
      }
    }

    // Update the observation identifier.
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) 
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  const double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  
  // The denominators also stay the same
  const double x_denom = 2 * std_landmark[0] * std_landmark[0];
  const double y_denom = 2 * std_landmark[1] * std_landmark[1];

  // Iterate through each particle
  for (int i = 0; i < num_particles; ++i) 
  {    
    // Calculate multi-variate Gaussian distribution of each observation
    double mvGd = 1.0;
    
    // For each observation
    for (int j = 0; j < observations.size(); ++j) 
    {      
      // Transform the observation point: vehicle coordinates → map coordinates
      double trans_obs_x, trans_obs_y;
      trans_obs_x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
      trans_obs_y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
      
      // Find nearest landmark
      vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
      vector<double> landmark_obs_dist (landmarks.size());
      
      for (int k = 0; k < landmarks.size(); ++k) 
      {
        // Down-size possible amount of landmarks
        // If in range → put in the distance vector for calculating nearest neighbor
        double landmark_part_dist = sqrt(pow(particles[i].x - landmarks[k].x_f, 2) + pow(particles[i].y - landmarks[k].y_f, 2));
        if (landmark_part_dist <= sensor_range) {
          landmark_obs_dist[k] = sqrt(pow(trans_obs_x - landmarks[k].x_f, 2) + pow(trans_obs_y - landmarks[k].y_f, 2));

        } 
        else 
        {
          // Do not set those to 0! As it will be interpreted as closest!
          landmark_obs_dist[k] = 999999.0;
        }
      }
      
      // Associate the observation point with its nearest landmark neighbor
      int min_pos = distance(landmark_obs_dist.begin(),min_element(landmark_obs_dist.begin(),landmark_obs_dist.end()));
      float nn_x = landmarks[min_pos].x_f;
      float nn_y = landmarks[min_pos].y_f;
      
      // Calculate multi-variate Gaussian distribution
      double x_diff = trans_obs_x - nn_x;
      double y_diff = trans_obs_y - nn_y;
      double b = ((x_diff * x_diff) / x_denom) + ((y_diff * y_diff) / y_denom);
      mvGd *= a * exp(-b);
    }
    
    // Update particle weights with combined multi-variate Gaussian distribution
    particles[i].weight = mvGd;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() 
{
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Vector for new particles
  vector<Particle> new_particles (num_particles);
  
  // Use discrete distribution to return particles by weight
  random_device rd;
  default_random_engine gen(rd());
  
  for (int i = 0; i < num_particles; ++i) 
  {
    discrete_distribution<int> index(weights.begin(), weights.end());
    new_particles[i] = particles[index(gen)];
  }
  
  // Replace old particles with the resampled particles
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  
  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") 
  {
    v = best.sense_x;
  } 
  else 
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}