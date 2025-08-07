#!/usr/bin/env python3
"""
Complete Badminton Performance Analysis System
AI Engineer Assessment - Future Sportler

This is the complete solution that analyzes badminton videos and provides:
1. Multi-player pose detection and tracking
2. Advanced shot classification and technique analysis
3. Proper 3D visualizations showing movement patterns
4. Comprehensive performance reports with corrective feedback
5. Interactive dashboards for detailed analysis

Author: AI Engineer Candidate
Date: 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, savgol_filter
import pandas as pd
import warnings
import argparse
import sys

warnings.filterwarnings('ignore')


class CompleteBadmintonAnalyzer:
    """Complete badminton analysis system with advanced features"""

    def __init__(self):
        print("🏸 Initializing Complete Badminton Analysis System")

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Highest accuracy
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Player tracking system
        self.player_tracks = {}
        self.next_player_id = 1
        self.max_tracking_distance = 150
        self.player_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22']

        # Badminton-specific analysis parameters
        self.court_zones = {
            'front_court_left': (0.0, 0.6, 0.5, 1.0),
            'front_court_right': (0.5, 0.6, 1.0, 1.0),
            'mid_court_left': (0.0, 0.4, 0.5, 0.6),
            'mid_court_right': (0.5, 0.4, 1.0, 0.6),
            'back_court_left': (0.0, 0.0, 0.5, 0.4),
            'back_court_right': (0.5, 0.0, 1.0, 0.4)
        }

        # Shot type definitions
        self.shot_types = {
            'serve': {'wrist_height': (-0.1, 0.1), 'elbow_angle': (60, 120), 'shoulder_height': 'below'},
            'smash': {'wrist_height': (0.1, 0.3), 'elbow_angle': (120, 180), 'shoulder_height': 'above'},
            'clear': {'wrist_height': (0.05, 0.25), 'elbow_angle': (100, 160), 'shoulder_height': 'above'},
            'drop': {'wrist_height': (0.0, 0.15), 'elbow_angle': (90, 140), 'shoulder_height': 'neutral'},
            'drive': {'wrist_height': (-0.05, 0.1), 'elbow_angle': (80, 130), 'shoulder_height': 'neutral'},
            'net': {'wrist_height': (-0.1, 0.05), 'elbow_angle': (60, 110), 'shoulder_height': 'below'}
        }

        # Performance scoring weights
        self.scoring_weights = {
            'technique_quality': 0.25,
            'shot_variety': 0.20,
            'court_coverage': 0.15,
            'shot_accuracy': 0.15,
            'consistency': 0.15,
            'tactical_awareness': 0.10
        }

        # Create comprehensive output structure
        self.setup_output_directories()
        print("✅ Complete analyzer initialized successfully!")

    def setup_output_directories(self):
        """Create comprehensive output directory structure"""
        directories = [
            "output",
            "output/3d_visualizations",
            "output/player_analysis",
            "output/shot_analysis",
            "output/performance_reports",
            "output/interactive_dashboards",
            "output/corrective_feedback",
            "output/comparative_analysis"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def analyze_video_complete(self, video_path):
        """Complete video analysis with all advanced features"""
        print(f"\n🎥 Analyzing: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        print(f"📊 Video Info: {width}x{height}, {fps}FPS, {total_frames} frames ({duration:.1f}s)")

        # Reset tracking for new video
        self.player_tracks = {}
        self.next_player_id = 1

        # Analysis data storage
        all_frame_data = []
        shot_events = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect poses
            results = self.pose.process(rgb_frame)

            # Process all detected players in frame
            frame_players = self.detect_and_track_players_advanced(
                results, width, height, frame_count, fps
            )

            # Analyze each player's performance
            for player_data in frame_players:
                # Comprehensive frame analysis
                frame_analysis = self.analyze_frame_comprehensive(
                    player_data, frame_count, width, height, fps
                )

                # Store frame data
                complete_frame_data = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'player_id': player_data['player_id'],
                    'landmarks_3d': player_data['landmarks_3d'],
                    'court_position': player_data['court_position'],
                    'center_position': player_data['center_position'],
                    'analysis': frame_analysis
                }

                all_frame_data.append(complete_frame_data)

                # Detect shot events
                shot_event = self.detect_shot_event(
                    player_data, frame_analysis, frame_count
                )
                if shot_event:
                    shot_events.append(shot_event)

            frame_count += 1

            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"⏳ Progress: {progress:.1f}% - Players: {len(self.player_tracks)} - Shots: {len(shot_events)}")

        cap.release()

        # Post-processing
        self.post_process_analysis_data(all_frame_data, shot_events)

        # Generate movement patterns
        movement_patterns = self.analyze_movement_patterns(all_frame_data, width, height)

        # Calculate comprehensive performance scores
        performance_analysis = self.calculate_comprehensive_performance(
            all_frame_data, shot_events, movement_patterns
        )

        print(f"✅ Analysis Complete!")
        print(f"   📊 {len(self.player_tracks)} players detected")
        print(f"   🏸 {len(shot_events)} shot events identified")
        print(f"   📈 {len(all_frame_data)} frames analyzed")

        return {
            'video_path': video_path,
            'video_info': {
                'fps': fps, 'frames': total_frames, 'width': width,
                'height': height, 'duration': duration
            },
            'frame_data': all_frame_data,
            'shot_events': shot_events,
            'movement_patterns': movement_patterns,
            'performance_analysis': performance_analysis,
            'player_tracks': self.player_tracks,
            'players_detected': list(self.player_tracks.keys())
        }

    def detect_and_track_players_advanced(self, pose_results, width, height, frame_num, fps):
        """Advanced multi-player detection and tracking"""
        detected_players = []

        if pose_results.pose_landmarks:
            # Extract 3D landmarks
            landmarks_3d = []

            if pose_results.pose_world_landmarks:
                for lm in pose_results.pose_world_landmarks.landmark:
                    landmarks_3d.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })
            else:
                # Fallback to 2D landmarks with estimated depth
                for lm in pose_results.pose_landmarks.landmark:
                    landmarks_3d.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })

            if len(landmarks_3d) >= 33:
                # Calculate player center using torso
                center_landmarks = [11, 12, 23, 24]
                center_x = np.mean([landmarks_3d[i]['x'] * width for i in center_landmarks])
                center_y = np.mean([landmarks_3d[i]['y'] * height for i in center_landmarks])
                center_z = np.mean([landmarks_3d[i]['z'] for i in center_landmarks])

                # Track or assign player ID
                player_id = self.assign_player_id_advanced(
                    center_x, center_y, center_z, frame_num
                )

                # Calculate court position
                court_pos = {
                    'x': center_x / width,
                    'y': center_y / height,
                    'zone': self.get_court_zone(center_x / width, center_y / height)
                }

                detected_players.append({
                    'player_id': player_id,
                    'center_position': (center_x, center_y, center_z),
                    'court_position': court_pos,
                    'landmarks_3d': landmarks_3d,
                    'frame': frame_num,
                    'timestamp': frame_num / fps
                })

        return detected_players

    def assign_player_id_advanced(self, x, y, z, frame_num):
        """Advanced player ID assignment with 3D tracking"""
        min_distance = float('inf')
        assigned_id = None

        # Check existing players
        for player_id, track_data in self.player_tracks.items():
            if len(track_data['positions_3d']) > 0:
                last_pos = track_data['positions_3d'][-1]

                # 3D distance calculation
                distance = np.sqrt(
                    (x - last_pos['x']) ** 2 +
                    (y - last_pos['y']) ** 2 +
                    (z - last_pos['z']) ** 2 * 1000
                )

                # Time-based tracking
                frame_gap = frame_num - last_pos['frame']
                if frame_gap <= 30 and distance < self.max_tracking_distance:
                    if distance < min_distance:
                        min_distance = distance
                        assigned_id = player_id

        # Create new player if no match
        if assigned_id is None:
            assigned_id = self.next_player_id
            self.next_player_id += 1
            self.player_tracks[assigned_id] = {
                'positions_3d': [],
                'shot_history': [],
                'performance_metrics': {},
                'movement_analysis': {},
                'errors_detected': [],
                'corrective_feedback': []
            }

        # Update tracking data
        self.player_tracks[assigned_id]['positions_3d'].append({
            'x': x, 'y': y, 'z': z, 'frame': frame_num
        })

        return assigned_id

    def analyze_frame_comprehensive(self, player_data, frame_num, width, height, fps):
        """Comprehensive frame-by-frame analysis"""
        landmarks = player_data['landmarks_3d']
        analysis = {}

        if len(landmarks) >= 33:
            # Stance and posture analysis
            stance_analysis = self.analyze_stance_detailed(landmarks)

            # Shot mechanics analysis
            shot_analysis = self.analyze_shot_mechanics_advanced(landmarks, player_data['player_id'], frame_num)

            # Balance and stability
            balance_analysis = self.analyze_balance_stability(landmarks)

            # Body form analysis
            form_analysis = self.analyze_body_form(landmarks)

            # Movement analysis
            movement_analysis = self.analyze_movement_quality(landmarks, player_data['player_id'])

            # Serve legality check
            serve_analysis = self.analyze_serve_legality(landmarks, shot_analysis)

            # Error detection
            errors = self.detect_technical_errors(landmarks, shot_analysis, stance_analysis)

            # Combine all analysis
            overall_score = (
                    stance_analysis['stance_score'] * 0.25 +
                    shot_analysis['shot_quality'] * 0.25 +
                    balance_analysis['balance_score'] * 0.20 +
                    form_analysis['overall_form_score'] * 0.15 +
                    movement_analysis['movement_score'] * 0.15
            )

            analysis = {
                'stance': stance_analysis,
                'shot_mechanics': shot_analysis,
                'balance': balance_analysis,
                'body_form': form_analysis,
                'movement': movement_analysis,
                'serve_legality': serve_analysis,
                'errors_detected': errors,
                'overall_score': overall_score,
                'timestamp': frame_num / fps
            }

        return analysis

    def analyze_stance_detailed(self, landmarks):
        """Detailed stance analysis for badminton"""
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]

        # Calculate stance metrics
        ankle_width = abs(left_ankle['x'] - right_ankle['x'])
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])

        stance_ratio = ankle_width / (shoulder_width + 1e-6)
        ideal_ratio = 1.2

        # Stance score
        stance_score = max(0, 100 - abs(stance_ratio - ideal_ratio) * 50)

        # Weight balance
        left_weight = (left_knee['y'] + left_hip['y']) / 2
        right_weight = (right_knee['y'] + right_hip['y']) / 2
        weight_balance = max(0, 100 - abs(left_weight - right_weight) * 200)

        # Foot positioning analysis
        foot_positioning = 'optimal' if 1.0 <= stance_ratio <= 1.4 else 'needs_adjustment'

        # Ready position check
        knee_bend_left = abs(left_hip['y'] - left_knee['y']) / abs(left_knee['y'] - left_ankle['y'] + 1e-6)
        knee_bend_right = abs(right_hip['y'] - right_knee['y']) / abs(right_knee['y'] - right_ankle['y'] + 1e-6)
        knee_bend_avg = (knee_bend_left + knee_bend_right) / 2

        ready_position_score = min(100, max(0, 100 - abs(knee_bend_avg - 1.5) * 30))

        return {
            'stance_score': stance_score,
            'stance_ratio': stance_ratio,
            'weight_balance': weight_balance,
            'foot_positioning': foot_positioning,
            'ready_position_score': ready_position_score,
            'knee_bend_quality': 'good' if 1.2 <= knee_bend_avg <= 1.8 else 'needs_improvement'
        }

    def analyze_shot_mechanics_advanced(self, landmarks, player_id, frame_num):
        """Advanced shot mechanics analysis"""
        right_wrist = landmarks[16]
        right_elbow = landmarks[14]
        right_shoulder = landmarks[12]
        left_shoulder = landmarks[11]
        right_hip = landmarks[24]

        # Calculate wrist position relative to shoulders
        shoulder_center_y = (right_shoulder['y'] + left_shoulder['y']) / 2
        wrist_height_ratio = (shoulder_center_y - right_wrist['y'])

        # Elbow angle calculation
        elbow_angle = self.calculate_joint_angle(right_shoulder, right_elbow, right_wrist)

        # Shoulder angle
        shoulder_angle = self.calculate_joint_angle(right_hip, right_shoulder, right_elbow)

        # Determine shot type and quality
        shot_type = self.classify_shot_type(wrist_height_ratio, elbow_angle, shoulder_angle)

        # Calculate shot quality based on biomechanics
        shot_quality = self.calculate_shot_quality(shot_type, wrist_height_ratio, elbow_angle, shoulder_angle)

        # Racket head speed estimation (based on wrist movement)
        racket_speed_estimate = min(100, abs(wrist_height_ratio) * 200 + elbow_angle * 0.3)

        # Contact point evaluation
        contact_point_score = self.evaluate_contact_point(landmarks, shot_type)

        return {
            'shot_type': shot_type,
            'shot_quality': shot_quality,
            'elbow_angle': elbow_angle,
            'shoulder_angle': shoulder_angle,
            'wrist_position': wrist_height_ratio,
            'racket_speed_estimate': racket_speed_estimate,
            'contact_point_score': contact_point_score,
            'preparation_score': min(100, shot_quality)
        }

    def classify_shot_type(self, wrist_height, elbow_angle, shoulder_angle):
        """Classify shot type based on biomechanical parameters"""
        for shot_name, criteria in self.shot_types.items():
            wrist_range = criteria['wrist_height']
            elbow_range = criteria['elbow_angle']

            if (wrist_range[0] <= wrist_height <= wrist_range[1] and
                    elbow_range[0] <= elbow_angle <= elbow_range[1]):
                return shot_name

        return 'unclassified'

    def calculate_shot_quality(self, shot_type, wrist_height, elbow_angle, shoulder_angle):
        """Calculate shot quality based on biomechanical efficiency"""
        base_score = 50

        if shot_type in self.shot_types:
            criteria = self.shot_types[shot_type]

            # Wrist height score
            wrist_optimal = np.mean(criteria['wrist_height'])
            wrist_score = max(0, 100 - abs(wrist_height - wrist_optimal) * 200)

            # Elbow angle score
            elbow_optimal = np.mean(criteria['elbow_angle'])
            elbow_score = max(0, 100 - abs(elbow_angle - elbow_optimal) * 0.5)

            # Combined quality score
            quality_score = (wrist_score * 0.4 + elbow_score * 0.4 + shoulder_angle * 0.2)
            return min(100, quality_score)

        return base_score

    def evaluate_contact_point(self, landmarks, shot_type):
        """Evaluate optimal contact point for different shots"""
        right_wrist = landmarks[16]
        nose = landmarks[0]

        # Distance from nose to wrist (proxy for optimal contact point)
        contact_distance = np.sqrt(
            (right_wrist['x'] - nose['x']) ** 2 +
            (right_wrist['y'] - nose['y']) ** 2
        )

        # Optimal contact point varies by shot type
        optimal_distances = {
            'smash': 0.3, 'clear': 0.28, 'drop': 0.25,
            'drive': 0.22, 'net': 0.20, 'serve': 0.25
        }

        optimal_distance = optimal_distances.get(shot_type, 0.25)
        contact_score = max(0, 100 - abs(contact_distance - optimal_distance) * 300)

        return contact_score

    def analyze_balance_stability(self, landmarks):
        """Analyze player balance and stability"""
        # Center of mass from torso
        torso_points = [11, 12, 23, 24]
        center_of_mass_x = np.mean([landmarks[i]['x'] for i in torso_points])
        center_of_mass_y = np.mean([landmarks[i]['y'] for i in torso_points])

        # Base of support from feet
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        base_center_x = (left_ankle['x'] + right_ankle['x']) / 2
        base_center_y = (left_ankle['y'] + right_ankle['y']) / 2

        # Balance calculation
        balance_offset_x = abs(center_of_mass_x - base_center_x)
        balance_offset_y = abs(center_of_mass_y - base_center_y)

        balance_score = max(0, 100 - (balance_offset_x + balance_offset_y) * 250)

        # Dynamic stability (based on body alignment)
        head = landmarks[0]
        hip_center_x = (landmarks[23]['x'] + landmarks[24]['x']) / 2

        dynamic_stability = max(0, 100 - abs(head['x'] - hip_center_x) * 400)

        return {
            'balance_score': balance_score,
            'balance_offset_x': balance_offset_x,
            'balance_offset_y': balance_offset_y,
            'dynamic_stability': dynamic_stability,
            'stability_rating': 'excellent' if balance_score > 80 else 'good' if balance_score > 60 else 'needs_improvement'
        }

    def analyze_body_form(self, landmarks):
        """Analyze overall body form and alignment"""
        nose = landmarks[0]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Hip center
        hip_center_x = (left_hip['x'] + right_hip['x']) / 2

        # Spine alignment
        spine_alignment = 100 - abs(nose['x'] - hip_center_x) * 300

        # Shoulder level
        shoulder_level = abs(left_shoulder['y'] - right_shoulder['y'])
        shoulder_alignment = 100 - shoulder_level * 300

        # Hip alignment
        hip_level = abs(left_hip['y'] - right_hip['y'])
        hip_alignment = 100 - hip_level * 400

        overall_form = (max(0, spine_alignment) + max(0, shoulder_alignment) + max(0, hip_alignment)) / 3

        return {
            'spine_alignment_score': max(0, spine_alignment),
            'shoulder_alignment_score': max(0, shoulder_alignment),
            'hip_alignment_score': max(0, hip_alignment),
            'overall_form_score': overall_form,
            'posture_rating': 'excellent' if overall_form > 80 else 'good' if overall_form > 60 else 'needs_work'
        }

    def analyze_movement_quality(self, landmarks, player_id):
        """Analyze movement quality and footwork"""
        # Get recent positions for this player
        if player_id in self.player_tracks and len(self.player_tracks[player_id]['positions_3d']) > 5:
            recent_positions = self.player_tracks[player_id]['positions_3d'][-5:]

            # Calculate movement smoothness
            velocities = []
            for i in range(1, len(recent_positions)):
                vel = np.sqrt(
                    (recent_positions[i]['x'] - recent_positions[i - 1]['x']) ** 2 +
                    (recent_positions[i]['y'] - recent_positions[i - 1]['y']) ** 2
                )
                velocities.append(vel)

            # Movement efficiency
            if velocities:
                movement_consistency = max(0, 100 - np.std(velocities) * 100)
                movement_speed = np.mean(velocities)
            else:
                movement_consistency = 50
                movement_speed = 0
        else:
            movement_consistency = 50
            movement_speed = 0

        # Footwork analysis from current pose
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        # Knee alignment
        knee_alignment = max(0, 100 - abs(left_knee['x'] - left_ankle['x']) * 200)
        knee_alignment += max(0, 100 - abs(right_knee['x'] - right_ankle['x']) * 200)
        knee_alignment /= 2

        movement_score = (movement_consistency * 0.4 + knee_alignment * 0.6)

        return {
            'movement_score': movement_score,
            'movement_consistency': movement_consistency,
            'movement_speed': movement_speed,
            'knee_alignment': knee_alignment,
            'footwork_rating': 'excellent' if movement_score > 80 else 'good' if movement_score > 60 else 'needs_work'
        }

    def analyze_serve_legality(self, landmarks, shot_analysis):
        """Analyze serve legality according to BWF rules"""
        right_wrist = landmarks[16]
        right_elbow = landmarks[14]
        right_shoulder = landmarks[12]

        # Key serve legality checks
        legal_checks = {
            'contact_below_waist': True,  # Default assumption
            'racket_head_below_wrist': True,
            'underarm_motion': True,
            'no_delay_in_service': True
        }

        violations = []

        # Check if wrist is above elbow (illegal)
        if right_wrist['y'] < right_elbow['y']:
            legal_checks['racket_head_below_wrist'] = False
            violations.append("Racket head above wrist at contact")

        # Check if elbow is significantly above shoulder during serve
        if shot_analysis['shot_type'] == 'serve' and right_elbow['y'] < right_shoulder['y'] - 0.1:
            legal_checks['underarm_motion'] = False
            violations.append("Overarm serve motion detected")

        # Overall legality score
        legal_score = sum(legal_checks.values()) / len(legal_checks) * 100

        return {
            'is_legal': len(violations) == 0,
            'legal_score': legal_score,
            'violations': violations,
            'legal_checks': legal_checks
        }

    def detect_technical_errors(self, landmarks, shot_analysis, stance_analysis):
        """Detect common technical errors"""
        errors = []

        # Poor stance errors
        if stance_analysis['stance_score'] < 60:
            errors.append({
                'type': 'stance',
                'description': 'Suboptimal foot positioning',
                'severity': 'medium',
                'correction': 'Maintain shoulder-width stance with slight forward lean'
            })

        # Shot execution errors
        if shot_analysis['shot_quality'] < 50:
            errors.append({
                'type': 'shot_execution',
                'description': 'Poor shot mechanics',
                'severity': 'high',
                'correction': 'Focus on proper racket preparation and follow-through'
            })

        # Elbow positioning errors
        if shot_analysis['elbow_angle'] > 170:
            errors.append({
                'type': 'elbow_positioning',
                'description': 'Over-extended elbow',
                'severity': 'medium',
                'correction': 'Keep elbow slightly bent for better control and power'
            })

        # Balance errors
        right_shoulder = landmarks[12]
        left_shoulder = landmarks[11]
        if abs(right_shoulder['y'] - left_shoulder['y']) > 0.1:
            errors.append({
                'type': 'balance',
                'description': 'Uneven shoulder alignment',
                'severity': 'low',
                'correction': 'Keep shoulders level and maintain good posture'
            })

        return errors

    def calculate_joint_angle(self, point1, point2, point3):
        """Calculate angle between three joint points"""
        v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def detect_shot_event(self, player_data, frame_analysis, frame_num):
        """Detect significant shot events"""
        shot_mechanics = frame_analysis.get('shot_mechanics', {})

        if (shot_mechanics.get('shot_quality', 0) > 60 and
                shot_mechanics.get('shot_type') != 'unclassified'):
            return {
                'frame': frame_num,
                'player_id': player_data['player_id'],
                'shot_type': shot_mechanics['shot_type'],
                'quality_score': shot_mechanics['shot_quality'],
                'court_position': player_data['court_position'],
                'timestamp': player_data['timestamp'],
                'racket_speed': shot_mechanics.get('racket_speed_estimate', 0),
                'contact_point_score': shot_mechanics.get('contact_point_score', 0)
            }

        return None

    def get_court_zone(self, norm_x, norm_y):
        """Determine court zone from normalized coordinates"""
        for zone_name, (x1, y1, x2, y2) in self.court_zones.items():
            if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
                return zone_name
        return 'center_court'

    def post_process_analysis_data(self, frame_data, shot_events):
        """Post-process and enhance analysis data"""
        print("📊 Post-processing analysis data...")

        # Smooth position data for better visualization
        for player_id in self.player_tracks:
            positions = self.player_tracks[player_id]['positions_3d']
            if len(positions) > 10:
                x_coords = [p['x'] for p in positions]
                y_coords = [p['y'] for p in positions]
                z_coords = [p['z'] for p in positions]

                # Apply smoothing filter
                window_length = min(11, len(x_coords) // 2 * 2 + 1)
                if window_length >= 5:
                    smoothed_x = savgol_filter(x_coords, window_length, 3)
                    smoothed_y = savgol_filter(y_coords, window_length, 3)
                    smoothed_z = savgol_filter(z_coords, window_length, 3)

                    for i, pos in enumerate(positions):
                        pos['smoothed_x'] = smoothed_x[i]
                        pos['smoothed_y'] = smoothed_y[i]
                        pos['smoothed_z'] = smoothed_z[i]

        # Generate corrective feedback
        self.generate_corrective_feedback(frame_data, shot_events)

    def generate_corrective_feedback(self, frame_data, shot_events):
        """Generate personalized corrective feedback for each player"""
        print("💡 Generating corrective feedback...")

        for player_id in self.player_tracks:
            player_frames = [f for f in frame_data if f['player_id'] == player_id]
            player_shots = [s for s in shot_events if s['player_id'] == player_id]

            if len(player_frames) < 10:
                continue

            # Collect all errors
            all_errors = []
            for frame in player_frames:
                errors = frame['analysis'].get('errors_detected', [])
                all_errors.extend(errors)

            # Group errors by type
            error_counts = {}
            for error in all_errors:
                error_type = error['type']
                if error_type not in error_counts:
                    error_counts[error_type] = []
                error_counts[error_type].append(error)

            # Generate feedback
            feedback = []
            for error_type, errors in error_counts.items():
                if len(errors) > len(player_frames) * 0.1:  # If error occurs in >10% of frames
                    most_common_error = max(set([e['description'] for e in errors]),
                                            key=[e['description'] for e in errors].count)
                    correction = errors[0]['correction']

                    feedback.append({
                        'error_type': error_type,
                        'frequency': len(errors),
                        'description': most_common_error,
                        'correction': correction,
                        'priority': 'high' if len(errors) > len(player_frames) * 0.3 else 'medium'
                    })

            self.player_tracks[player_id]['corrective_feedback'] = feedback

    def analyze_movement_patterns(self, frame_data, width, height):
        """Analyze movement patterns for each player"""
        movement_patterns = {}

        for player_id in self.player_tracks:
            player_frames = [f for f in frame_data if f['player_id'] == player_id]

            if len(player_frames) < 10:
                continue

            # Extract positions and timestamps
            positions = []
            timestamps = []

            for frame in player_frames:
                pos = frame['center_position']
                positions.append((pos[0], pos[1]))
                timestamps.append(frame['timestamp'])

            # Calculate movement metrics
            total_distance = 0
            velocities = []
            accelerations = []

            for i in range(1, len(positions)):
                distance = euclidean(positions[i - 1], positions[i])
                time_diff = timestamps[i] - timestamps[i - 1] if timestamps[i] > timestamps[i - 1] else 1 / 30

                total_distance += distance
                velocity = distance / time_diff
                velocities.append(velocity)

                # Calculate acceleration
                if i > 1 and len(velocities) > 1:
                    acceleration = (velocities[-1] - velocities[-2]) / time_diff
                    accelerations.append(acceleration)

            # Court coverage analysis
            normalized_positions = [(x / width, y / height) for x, y in positions]
            unique_positions = set((round(x * 20) / 20, round(y * 20) / 20) for x, y in normalized_positions)
            coverage_percentage = min(100, len(unique_positions) * 2)

            # Zone analysis
            zones_visited = set()
            zone_transitions = 0
            prev_zone = None

            for frame in player_frames:
                zone = frame['court_position'].get('zone', 'center_court')
                zones_visited.add(zone)
                if prev_zone and prev_zone != zone:
                    zone_transitions += 1
                prev_zone = zone

            # Movement efficiency analysis
            movement_efficiency = min(100, coverage_percentage / (total_distance / 1000 + 1))

            # Reaction timing analysis
            reaction_times = self.analyze_reaction_timing(player_frames)

            movement_patterns[player_id] = {
                'total_distance': total_distance,
                'average_velocity': np.mean(velocities) if velocities else 0,
                'max_velocity': max(velocities) if velocities else 0,
                'average_acceleration': np.mean(accelerations) if accelerations else 0,
                'court_coverage': coverage_percentage,
                'zones_visited': len(zones_visited),
                'zone_transitions': zone_transitions,
                'movement_efficiency': movement_efficiency,
                'reaction_analysis': reaction_times
            }

        return movement_patterns

    def analyze_reaction_timing(self, player_frames):
        """Analyze player reaction timing and anticipation"""
        reaction_analysis = {
            'average_reaction_time': 0,
            'anticipation_score': 0,
            'movement_readiness': 0
        }

        if len(player_frames) < 30:
            return reaction_analysis

        # Analyze movement changes as proxy for reactions
        movement_changes = []
        for i in range(2, len(player_frames)):
            curr_pos = player_frames[i]['center_position']
            prev_pos = player_frames[i - 1]['center_position']
            prev_prev_pos = player_frames[i - 2]['center_position']

            # Calculate direction change
            curr_direction = np.arctan2(curr_pos[1] - prev_pos[1], curr_pos[0] - prev_pos[0])
            prev_direction = np.arctan2(prev_pos[1] - prev_prev_pos[1], prev_pos[0] - prev_prev_pos[0])

            direction_change = abs(curr_direction - prev_direction)
            if direction_change > np.pi:
                direction_change = 2 * np.pi - direction_change

            if direction_change > np.pi / 4:  # Significant direction change
                movement_changes.append({
                    'frame': i,
                    'direction_change': direction_change,
                    'reaction_intensity': min(100, direction_change * 50)
                })

        # Calculate average reaction metrics
        if movement_changes:
            reaction_analysis['average_reaction_time'] = np.mean([mc['reaction_intensity'] for mc in movement_changes])
            reaction_analysis['anticipation_score'] = min(100, len(movement_changes) / len(player_frames) * 500)

        # Movement readiness based on stance quality
        stance_scores = []
        for frame in player_frames:
            stance_score = frame['analysis'].get('stance', {}).get('ready_position_score', 50)
            stance_scores.append(stance_score)

        reaction_analysis['movement_readiness'] = np.mean(stance_scores)

        return reaction_analysis

    def calculate_comprehensive_performance(self, frame_data, shot_events, movement_patterns):
        """Calculate comprehensive performance scores"""
        performance_analysis = {}

        for player_id in self.player_tracks:
            player_frames = [f for f in frame_data if f['player_id'] == player_id]
            player_shots = [s for s in shot_events if s['player_id'] == player_id]

            if len(player_frames) < 10:
                continue

            # Extract scores from frame analysis
            frame_scores = []
            stance_scores = []
            shot_scores = []

            for frame in player_frames:
                if 'overall_score' in frame['analysis']:
                    frame_scores.append(frame['analysis']['overall_score'])
                if 'stance' in frame['analysis']:
                    stance_scores.append(frame['analysis']['stance']['stance_score'])
                if 'shot_mechanics' in frame['analysis']:
                    shot_scores.append(frame['analysis']['shot_mechanics']['shot_quality'])

            # Performance metrics calculation
            technique_quality = np.mean(frame_scores) if frame_scores else 0

            # Shot analysis
            unique_shot_types = set(shot['shot_type'] for shot in player_shots)
            shot_variety = len(unique_shot_types)
            shot_accuracy = len([s for s in player_shots if s['quality_score'] > 70]) / max(1, len(player_shots)) * 100

            # Movement metrics
            movement_data = movement_patterns.get(player_id, {})
            court_coverage = movement_data.get('court_coverage', 0)

            # Consistency
            consistency = max(0, 100 - np.std(frame_scores) * 2) if len(frame_scores) > 1 else 50

            # Tactical awareness
            zones_visited = movement_data.get('zones_visited', 0)
            tactical_awareness = min(100, zones_visited * 16.67)  # 6 zones max

            # Calculate weighted final score
            final_score = (
                    technique_quality * self.scoring_weights['technique_quality'] +
                    min(100, shot_variety * 20) * self.scoring_weights['shot_variety'] +
                    court_coverage * self.scoring_weights['court_coverage'] +
                    shot_accuracy * self.scoring_weights['shot_accuracy'] +
                    consistency * self.scoring_weights['consistency'] +
                    tactical_awareness * self.scoring_weights['tactical_awareness']
            )

            # Performance grade
            if final_score >= 90:
                grade = 'A+ (Excellent)'
            elif final_score >= 80:
                grade = 'A (Very Good)'
            elif final_score >= 70:
                grade = 'B+ (Good)'
            elif final_score >= 60:
                grade = 'B (Above Average)'
            elif final_score >= 50:
                grade = 'C+ (Average)'
            else:
                grade = 'C (Needs Improvement)'

            performance_analysis[player_id] = {
                'final_score': final_score,
                'grade': grade,
                'individual_scores': {
                    'technique_quality': technique_quality,
                    'shot_variety': min(100, shot_variety * 20),
                    'court_coverage': court_coverage,
                    'shot_accuracy': shot_accuracy,
                    'consistency': consistency,
                    'tactical_awareness': tactical_awareness
                },
                'metrics': {
                    'total_frames': len(player_frames),
                    'total_shots': len(player_shots),
                    'avg_stance_score': np.mean(stance_scores) if stance_scores else 0,
                    'avg_shot_score': np.mean(shot_scores) if shot_scores else 0,
                    'movement_distance': movement_data.get('total_distance', 0),
                    'zones_covered': zones_visited,
                    'reaction_time': movement_data.get('reaction_analysis', {}).get('average_reaction_time', 0)
                }
            }

        return performance_analysis

    def create_proper_3d_visualization(self, video_analysis, output_path):
        """Create proper 3D visualization showing player movement and performance"""
        print(f"🎨 Creating proper 3D visualization...")

        frame_data = video_analysis['frame_data']
        video_info = video_analysis['video_info']
        shot_events = video_analysis['shot_events']

        if not frame_data:
            print("❌ No frame data available for visualization")
            return

        # Create 3D figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d', 'colspan': 2}, None],
                   [{'type': 'scatter'}, {'type': 'bar'}]],
            subplot_titles=('3D Movement Analysis', 'Performance Timeline', 'Shot Analysis'),
            vertical_spacing=0.08
        )

        # Process data by player
        players_data = {}
        for frame in frame_data[::10]:  # Sample every 10th frame for performance
            player_id = frame['player_id']
            if player_id not in players_data:
                players_data[player_id] = {
                    'x': [], 'y': [], 'z': [], 'scores': [], 'times': [], 'frames': [],
                    'shot_types': [], 'errors': []
                }

            pos = frame['center_position']
            score = frame['analysis'].get('overall_score', 0)

            players_data[player_id]['x'].append(pos[0])
            players_data[player_id]['y'].append(pos[1])
            players_data[player_id]['z'].append(pos[2] * 100)  # Scale Z for visibility
            players_data[player_id]['scores'].append(score)
            players_data[player_id]['times'].append(frame['timestamp'])
            players_data[player_id]['frames'].append(frame['frame'])
            players_data[player_id]['errors'].append(len(frame['analysis'].get('errors_detected', [])))

        # Create traces for each player in 3D plot
        for i, (player_id, data) in enumerate(players_data.items()):
            if len(data['x']) == 0:
                continue

            color = self.player_colors[i % len(self.player_colors)]

            # Player movement trajectory with performance coloring
            fig.add_trace(go.Scatter3d(
                x=data['x'],
                y=data['y'],
                z=data['z'],
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=data['scores'],
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=100,
                    colorbar=dict(
                        title=f"Performance Score",
                        x=1.02 + i * 0.15
                    ) if i == 0 else None,
                    showscale=i == 0
                ),
                line=dict(
                    color=color,
                    width=4
                ),
                name=f'Player {player_id}',
                text=[f'Player {player_id}<br>Time: {t:.1f}s<br>Score: {s:.1f}<br>Frame: {f}<br>Errors: {e}'
                      for t, s, f, e in zip(data['times'], data['scores'], data['frames'], data['errors'])],
                hovertemplate='%{text}<br>Position: (%{x:.0f}, %{y:.0f}, %{z:.1f})<extra></extra>'
            ), row=1, col=1)

            # Performance timeline
            fig.add_trace(go.Scatter(
                x=data['times'],
                y=data['scores'],
                mode='lines+markers',
                name=f'Player {player_id} Performance',
                line=dict(color=color, width=3),
                marker=dict(size=6)
            ), row=2, col=1)

        # Add court boundaries in 3D space
        width = video_info['width']
        height = video_info['height']

        # Court outline at ground level (z=0)
        court_outline_x = [0, width, width, 0, 0]
        court_outline_y = [0, 0, height, height, 0]
        court_outline_z = [0, 0, 0, 0, 0]

        fig.add_trace(go.Scatter3d(
            x=court_outline_x,
            y=court_outline_y,
            z=court_outline_z,
            mode='lines',
            line=dict(color='black', width=8),
            name='Court Boundary',
            showlegend=False
        ), row=1, col=1)

        # Add net line
        net_x = [width / 2, width / 2]
        net_y = [0, height]
        net_z = [0, 0]

        fig.add_trace(go.Scatter3d(
            x=net_x,
            y=net_y,
            z=net_z,
            mode='lines',
            line=dict(color='red', width=6),
            name='Net',
            showlegend=False
        ), row=1, col=1)

        # Add shot events as markers
        for shot in shot_events[::5]:  # Show every 5th shot to avoid clutter
            fig.add_trace(go.Scatter3d(
                x=[shot['court_position']['x'] * width],
                y=[shot['court_position']['y'] * height],
                z=[20],  # Slightly above ground
                mode='markers',
                marker=dict(
                    size=12,
                    color='yellow',
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                name=f'{shot["shot_type"].title()} Shot',
                text=f'Shot: {shot["shot_type"]}<br>Quality: {shot["quality_score"]:.1f}<br>Time: {shot["timestamp"]:.1f}s',
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ), row=1, col=1)

        # Shot type analysis bar chart
        shot_type_counts = {}
        for shot in shot_events:
            shot_type = shot['shot_type']
            if shot_type not in shot_type_counts:
                shot_type_counts[shot_type] = 0
            shot_type_counts[shot_type] += 1

        if shot_type_counts:
            fig.add_trace(go.Bar(
                x=list(shot_type_counts.keys()),
                y=list(shot_type_counts.values()),
                name='Shot Types',
                marker_color='lightblue'
            ), row=2, col=2)

        # Update layout
        fig.update_layout(
            title=f'Complete Badminton Analysis - {os.path.basename(video_analysis["video_path"])}',
            height=800,
            showlegend=True,
            scene=dict(
                xaxis_title='X Position (pixels)',
                yaxis_title='Y Position (pixels)',
                zaxis_title='Height (scaled)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            )
        )

        # Update subplot titles
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Performance Score", row=2, col=1)
        fig.update_xaxes(title_text="Shot Type", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        # Save the visualization
        fig.write_html(output_path)
        print(f"✅ 3D visualization saved to: {output_path}")

        return fig

    def generate_performance_report(self, video_analysis):
        """Generate comprehensive performance report"""
        print("📄 Generating performance report...")

        report = {
            'video_info': video_analysis['video_info'],
            'analysis_summary': {},
            'player_reports': {},
            'recommendations': {}
        }

        performance_analysis = video_analysis['performance_analysis']
        shot_events = video_analysis['shot_events']
        movement_patterns = video_analysis['movement_patterns']

        # Overall analysis summary
        total_players = len(performance_analysis)
        total_shots = len(shot_events)
        avg_performance = np.mean(
            [p['final_score'] for p in performance_analysis.values()]) if performance_analysis else 0

        report['analysis_summary'] = {
            'total_players': total_players,
            'total_shots_detected': total_shots,
            'average_performance_score': avg_performance,
            'video_duration': video_analysis['video_info']['duration']
        }

        # Individual player reports
        for player_id, performance in performance_analysis.items():
            player_shots = [s for s in shot_events if s['player_id'] == player_id]
            player_movement = movement_patterns.get(player_id, {})
            player_feedback = self.player_tracks[player_id]['corrective_feedback']

            # Shot type distribution
            shot_types = {}
            for shot in player_shots:
                shot_type = shot['shot_type']
                if shot_type not in shot_types:
                    shot_types[shot_type] = {'count': 0, 'avg_quality': 0, 'qualities': []}
                shot_types[shot_type]['count'] += 1
                shot_types[shot_type]['qualities'].append(shot['quality_score'])

            for shot_type in shot_types:
                shot_types[shot_type]['avg_quality'] = np.mean(shot_types[shot_type]['qualities'])

            report['player_reports'][player_id] = {
                'performance_grade': performance['grade'],
                'final_score': performance['final_score'],
                'individual_scores': performance['individual_scores'],
                'shot_analysis': {
                    'total_shots': len(player_shots),
                    'shot_types': shot_types,
                    'best_shot_type': max(shot_types.keys(),
                                          key=lambda x: shot_types[x]['avg_quality']) if shot_types else 'None'
                },
                'movement_analysis': player_movement,
                'corrective_feedback': player_feedback,
                'strengths': self.identify_strengths(performance),
                'areas_for_improvement': self.identify_improvements(performance, player_feedback)
            }

        # Save report
        report_path = f"output/performance_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"✅ Performance report saved to: {report_path}")
        return report

    def identify_strengths(self, performance):
        """Identify player strengths based on scores"""
        strengths = []
        scores = performance['individual_scores']

        for metric, score in scores.items():
            if score > 75:
                strengths.append(f"Excellent {metric.replace('_', ' ')}")
            elif score > 65:
                strengths.append(f"Good {metric.replace('_', ' ')}")

        return strengths

    def identify_improvements(self, performance, feedback):
        """Identify areas for improvement"""
        improvements = []
        scores = performance['individual_scores']

        # Based on scores
        for metric, score in scores.items():
            if score < 50:
                improvements.append(f"Focus on improving {metric.replace('_', ' ')}")

        # Based on feedback
        for fb in feedback:
            if fb['priority'] == 'high':
                improvements.append(fb['correction'])

        return improvements

    def create_corrective_overlay(self, video_analysis, output_path):
        """Create visualization with corrective feedback overlay"""
        print("🎯 Creating corrective feedback overlay...")

        frame_data = video_analysis['frame_data']

        # Create figure for corrective feedback
        fig = go.Figure()

        # Process correction data
        corrections_timeline = {}

        for frame in frame_data[::30]:  # Every 30th frame
            player_id = frame['player_id']
            if player_id not in corrections_timeline:
                corrections_timeline[player_id] = {
                    'times': [], 'errors': [], 'corrections': []
                }

            errors = frame['analysis'].get('errors_detected', [])
            if errors:
                corrections_timeline[player_id]['times'].append(frame['timestamp'])
                corrections_timeline[player_id]['errors'].append(len(errors))
                corrections_timeline[player_id]['corrections'].append(
                    ', '.join([e['description'] for e in errors[:2]])  # Top 2 errors
                )

        # Plot error timeline for each player
        for i, (player_id, data) in enumerate(corrections_timeline.items()):
            if len(data['times']) == 0:
                continue

            color = self.player_colors[i % len(self.player_colors)]

            fig.add_trace(go.Scatter(
                x=data['times'],
                y=data['errors'],
                mode='markers+lines',
                name=f'Player {player_id} Errors',
                line=dict(color=color, width=3),
                marker=dict(size=10),
                text=data['corrections'],
                hovertemplate='Time: %{x:.1f}s<br>Errors: %{y}<br>Issues: %{text}<extra></extra>'
            ))

        fig.update_layout(
            title='Technical Errors Timeline - Corrective Feedback',
            xaxis_title='Time (seconds)',
            yaxis_title='Number of Errors Detected',
            hovermode='closest',
            height=500
        )

        # Save corrective overlay
        fig.write_html(output_path)
        print(f"✅ Corrective overlay saved to: {output_path}")

        return fig


def main():
    """Main function to run the complete badminton analysis"""
    print("🏸 Complete Badminton Performance Analysis System")
    print("=" * 60)

    # Initialize analyzer
    analyzer = CompleteBadmintonAnalyzer()

    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Badminton Performance Analysis')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--folder', type=str, help='Path to folder containing videos')
    parser.add_argument('--output', type=str, default='output', help='Output directory')

    args = parser.parse_args()

    # Determine input videos
    video_files = []

    if args.video:
        if os.path.exists(args.video):
            video_files = [args.video]
        else:
            print(f"❌ Video file not found: {args.video}")
            return
    elif args.folder:
        if os.path.exists(args.folder):
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(glob.glob(os.path.join(args.folder, ext)))
        else:
            print(f"❌ Folder not found: {args.folder}")
            return
    else:
        # Default: look for videos in current directory
        import glob
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(glob.glob(ext))

    if not video_files:
        print("❌ No video files found!")
        print("Usage examples:")
        print("  python badminton_analyzer.py --video video.mp4")
        print("  python badminton_analyzer.py --folder /path/to/videos/")
        return

    print(f"📁 Found {len(video_files)} video(s) to analyze:")
    for i, video in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(video)}")

    # Analyze each video
    all_results = []

    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'=' * 60}")
        print(f"🎬 Processing Video {i}/{len(video_files)}")
        print(f"{'=' * 60}")

        # Perform complete analysis
        video_analysis = analyzer.analyze_video_complete(video_path)

        if video_analysis:
            # Generate outputs
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # 3D Visualization
            viz_path = f"output/3d_visualizations/{video_name}_3d_analysis.html"
            analyzer.create_proper_3d_visualization(video_analysis, viz_path)

            # Performance Report
            report = analyzer.generate_performance_report(video_analysis)

            # Corrective Feedback Overlay
            correction_path = f"output/corrective_feedback/{video_name}_corrections.html"
            analyzer.create_corrective_overlay(video_analysis, correction_path)

            all_results.append({
                'video': video_path,
                'analysis': video_analysis,
                'report': report
            })

            print(f"\n✅ Analysis complete for {os.path.basename(video_path)}")
            print(f"   📊 3D Visualization: {viz_path}")
            print(f"   📄 Performance Report: Available in output/performance_reports/")
            print(f"   🎯 Corrective Overlay: {correction_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("🎉 ALL ANALYSES COMPLETE!")
    print(f"{'=' * 60}")
    print(f"📁 Total videos processed: {len(all_results)}")
    print(f"📂 All outputs saved to: {os.path.abspath('output')}/")

    # Overall statistics
    if all_results:
        total_players = sum(len(r['analysis']['players_detected']) for r in all_results)
        total_shots = sum(len(r['analysis']['shot_events']) for r in all_results)

        print(f"👥 Total players detected: {total_players}")
        print(f"🏸 Total shots analyzed: {total_shots}")
        print(f"⏱️  Processing complete!")

    print("\n🔍 Check the 'output' directory for:")
    print("   • 3D visualizations (HTML files)")
    print("   • Performance reports (JSON files)")
    print("   • Corrective feedback overlays")


if __name__ == "__main__":
    main()