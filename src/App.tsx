/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { 
  Shield, 
  Activity, 
  Database, 
  Cpu, 
  AlertTriangle, 
  Settings, 
  ChevronRight, 
  Box, 
  Zap,
  Globe,
  Wind,
  Code2,
  Terminal,
  Layers,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';

// --- Constants & Mock Data ---

const PYTHON_TGNN_CODE = `
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, TemporalConv
from torch_geometric.data import TemporalData

class SatelliteTGNN(nn.Module):
    """
    Temporal Graph Neural Network for Satellite Conjunction Prediction.
    Complexity: O(n) where n is the number of active proximities.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SatelliteTGNN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4)
        self.temp_conv = TemporalConv(hidden_channels * 4, hidden_channels * 4, kernel_size=3)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1)
        self.fc = nn.Linear(out_channels, 1) # Probability of collision
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, t):
        # Message passing with spatial attention
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        
        # Temporal feature extraction
        x = self.temp_conv(x)
        
        # Final spatial refinement
        x = self.conv2(x, edge_index, edge_attr)
        
        # Collision probability logit
        prob = self.sigmoid(self.fc(x))
        return prob

# Neo4j Schema:
# (:Satellite {id, norad_id, mass, drag_coeff})
# (:Debris {id, norad_id, size_est})
# [:PROXIMITY {distance, rel_velocity, time_stamp, covariance_matrix}]
`;

const PYTHON_VAE_CODE = `
import torch
import torch.nn as nn

class TurbineVAE(nn.Module):
    """
    Variational Autoencoder for Latent Anomaly Detection in Turbine Telemetry.
    """
    def __init__(self, input_dim, latent_dim):
        super(TurbineVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
`;

const HARDWARE_ANALYSIS = {
  edge_processing: [
    { component: "Compute Unit", spec: "NVIDIA Jetson Orin AGX (275 TOPS)", reason: "Real-time TGNN inference at the edge." },
    { component: "Memory", spec: "64GB LPDDR5", reason: "High-bandwidth graph state storage." },
    { component: "Networking", spec: "10GbE / SpaceWire", reason: "Kafka streaming ingestion from ground/space sensors." },
    { component: "Power", spec: "15W-60W Configurable", reason: "Optimized for orbital or atmospheric deployment." }
  ]
};

const MOCK_CONJUNCTIONS = [
  { id: 'C-1024', sat1: 'Starlink-2341', sat2: 'Cosmos-2251 (Debris)', distance: 0.42, probability: 0.08, time: 'T+14:20' },
  { id: 'C-1025', sat1: 'OneWeb-042', sat2: 'Fengyun-1C (Debris)', distance: 1.2, probability: 0.01, time: 'T+22:10' },
  { id: 'C-1026', sat1: 'ISS', sat2: 'Unknown Object', distance: 5.4, probability: 0.0001, time: 'T+45:00' },
];

// --- Components ---

const CodeBlock = ({ code, title }: { code: string, title: string }) => (
  <div className="bg-[#0a0a0c] border border-white/10 rounded-lg overflow-hidden font-mono text-sm">
    <div className="bg-white/5 px-4 py-2 border-b border-white/10 flex justify-between items-center">
      <span className="text-white/60 flex items-center gap-2">
        <Terminal size={14} /> {title}
      </span>
      <div className="flex gap-1.5">
        <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50" />
        <div className="w-2.5 h-2.5 rounded-full bg-green-500/50" />
      </div>
    </div>
    <pre className="p-4 overflow-x-auto text-blue-400">
      <code>{code}</code>
    </pre>
  </div>
);

const StatCard = ({ icon: Icon, label, value, color }: any) => (
  <div className="bg-white/5 border border-white/10 p-4 rounded-xl">
    <div className="flex items-center gap-3 mb-2">
      <div className={`p-2 rounded-lg ${color} bg-opacity-20`}>
        <Icon size={18} className={color.replace('bg-', 'text-')} />
      </div>
      <span className="text-white/40 text-sm font-medium">{label}</span>
    </div>
    <div className="text-2xl font-bold text-white">{value}</div>
  </div>
);

export default function App() {
  const [activeTab, setActiveTab] = useState('ssa');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [chartData, setChartData] = useState<any[]>([]);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    
    // Generate mock real-time data
    const interval = setInterval(() => {
      setChartData(prev => {
        const newData = [...prev, {
          time: new Date().toLocaleTimeString(),
          prob: Math.random() * 0.1,
          flux: 150 + Math.random() * 20
        }].slice(-20);
        return newData;
      });
    }, 2000);

    return () => {
      clearInterval(timer);
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#050507] text-white font-sans selection:bg-blue-500/30">
      {/* Sidebar */}
      <div className="fixed left-0 top-0 bottom-0 w-64 bg-[#0a0a0c] border-r border-white/5 z-50 hidden lg:flex flex-col">
        <div className="p-6 flex items-center gap-3 border-b border-white/5">
          <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-600/20">
            <Box className="text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-tight">ASTRAEUS</h1>
            <p className="text-[10px] text-white/40 uppercase tracking-widest font-semibold">Aerospace Systems</p>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {[
            { id: 'ssa', icon: Globe, label: 'SSA Monitor' },
            { id: 'tgnn', icon: Layers, label: 'TGNN Architecture' },
            { id: 'vae', icon: Wind, label: 'Predictive Maint.' },
            { id: 'hardware', icon: Cpu, label: 'Hardware Specs' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                activeTab === item.id 
                ? 'bg-blue-600/10 text-blue-400 border border-blue-600/20' 
                : 'text-white/40 hover:text-white hover:bg-white/5'
              }`}
            >
              <item.icon size={18} />
              <span className="font-medium">{item.label}</span>
              {activeTab === item.id && <ChevronRight size={14} className="ml-auto" />}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-white/5">
          <div className="bg-white/5 rounded-lg p-3 flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-purple-600 flex items-center justify-center text-xs font-bold">
              SE
            </div>
            <div className="flex-1 overflow-hidden">
              <p className="text-sm font-medium truncate">S. Emani</p>
              <p className="text-[10px] text-white/40 truncate">Lead Architect</p>
            </div>
            <Settings size={16} className="text-white/40" />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="lg:ml-64 p-8">
        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">
              {activeTab === 'ssa' && 'Space Situational Awareness'}
              {activeTab === 'tgnn' && 'Temporal Graph Neural Network'}
              {activeTab === 'vae' && 'Turbine Predictive Maintenance'}
              {activeTab === 'hardware' && 'Hardware Requirements'}
            </h2>
            <p className="text-white/40 mt-1">
              {activeTab === 'ssa' && 'Real-time satellite conjunction prediction and orbit monitoring.'}
              {activeTab === 'tgnn' && 'Deep learning architecture for dynamic proximity modeling.'}
              {activeTab === 'vae' && 'Latent anomaly detection for atmospheric flight telemetry.'}
              {activeTab === 'hardware' && 'Edge processing analysis for production-ready deployment.'}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right hidden sm:block">
              <p className="text-[10px] text-white/40 uppercase tracking-widest font-bold">System Status</p>
              <div className="flex items-center gap-2 text-green-400 font-mono text-sm">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                NOMINAL
              </div>
            </div>
            <div className="bg-white/5 border border-white/10 px-4 py-2 rounded-lg font-mono text-sm">
              {currentTime.toISOString().split('T')[1].split('.')[0]} UTC
            </div>
          </div>
        </header>

        <AnimatePresence mode="wait">
          {activeTab === 'ssa' && (
            <motion.div
              key="ssa"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard icon={Globe} label="Active Objects" value="24,512" color="bg-blue-500" />
                <StatCard icon={AlertTriangle} label="Critical Alerts" value="3" color="bg-red-500" />
                <StatCard icon={Zap} label="Solar Flux Index" value="164.2" color="bg-yellow-500" />
                <StatCard icon={Activity} label="TGNN Inference" value="12ms" color="bg-green-500" />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-white/5 border border-white/10 rounded-2xl p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="font-bold flex items-center gap-2">
                      <Activity size={18} className="text-blue-400" />
                      Collision Probability Trends
                    </h3>
                    <div className="flex gap-2">
                      <span className="px-2 py-1 bg-blue-500/10 text-blue-400 text-[10px] rounded border border-blue-500/20">LIVE FEED</span>
                    </div>
                  </div>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={chartData}>
                        <defs>
                          <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                        <XAxis dataKey="time" stroke="#ffffff40" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#ffffff40" fontSize={10} tickLine={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0a0a0c', border: '1px solid #ffffff10', borderRadius: '8px' }}
                          itemStyle={{ color: '#3b82f6' }}
                        />
                        <Area type="monotone" dataKey="prob" stroke="#3b82f6" fillOpacity={1} fill="url(#colorProb)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                  <h3 className="font-bold mb-4 flex items-center gap-2">
                    <AlertTriangle size={18} className="text-red-400" />
                    Active Conjunctions
                  </h3>
                  <div className="space-y-4">
                    {MOCK_CONJUNCTIONS.map(item => (
                      <div key={item.id} className="p-4 bg-white/5 rounded-xl border border-white/5 hover:border-white/10 transition-colors">
                        <div className="flex justify-between items-start mb-2">
                          <span className="text-xs font-mono text-white/40">{item.id}</span>
                          <span className={`text-[10px] px-2 py-0.5 rounded ${item.probability > 0.05 ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                            {(item.probability * 100).toFixed(1)}% RISK
                          </span>
                        </div>
                        <div className="text-sm font-medium mb-1">{item.sat1}</div>
                        <div className="text-xs text-white/40 mb-3">vs {item.sat2}</div>
                        <div className="flex items-center justify-between text-[10px] font-mono">
                          <span className="text-blue-400">DIST: {item.distance}km</span>
                          <span className="text-white/40">{item.time}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <button className="w-full mt-4 py-2 text-xs text-white/40 hover:text-white transition-colors border border-dashed border-white/10 rounded-lg">
                    VIEW ALL EVENTS
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'tgnn' && (
            <motion.div
              key="tgnn"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="grid grid-cols-1 xl:grid-cols-2 gap-8"
            >
              <div className="space-y-6">
                <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <Code2 size={20} className="text-blue-400" />
                    TGNN Model Implementation
                  </h3>
                  <p className="text-white/60 text-sm mb-6 leading-relaxed">
                    The Temporal Graph Neural Network utilizes message-passing layers in PyTorch Geometric to calculate collision probabilities with O(n) complexity. It models satellite proximities as dynamic edges in a Neo4j graph database.
                  </p>
                  <CodeBlock title="tgnn_model.py" code={PYTHON_TGNN_CODE} />
                </div>
                
                <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <Info size={20} className="text-purple-400" />
                    Explainability Layer (SHAP)
                  </h3>
                  <div className="space-y-4">
                    <div className="p-4 bg-white/5 rounded-xl border border-white/5">
                      <p className="text-sm font-medium mb-2">Feature Importance Breakdown</p>
                      <div className="space-y-3">
                        {[
                          { label: 'Relative Velocity', value: 85, color: 'bg-blue-500' },
                          { label: 'Solar Flux Impact', value: 62, color: 'bg-purple-500' },
                          { label: 'Covariance Growth', value: 45, color: 'bg-indigo-500' },
                          { label: 'Orbital Inclination', value: 28, color: 'bg-slate-500' },
                        ].map(f => (
                          <div key={f.label}>
                            <div className="flex justify-between text-[10px] mb-1 uppercase tracking-wider text-white/40">
                              <span>{f.label}</span>
                              <span>{f.value}%</span>
                            </div>
                            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                              <motion.div 
                                initial={{ width: 0 }}
                                animate={{ width: `${f.value}%` }}
                                className={`h-full ${f.color}`}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <p className="text-xs text-white/40 italic">
                      * SHAP values are used to justify recommended avoidance maneuvers to mission control by quantifying the contribution of each orbital parameter to the collision probability.
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                  <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                    <Database size={20} className="text-yellow-400" />
                    Data Ingestion Pipeline
                  </h3>
                  <div className="relative pl-8 space-y-8 before:absolute before:left-3 before:top-2 before:bottom-2 before:w-px before:bg-white/10">
                    {[
                      { title: 'Apache Kafka Ingestion', desc: 'Streaming TLE (Two-Line Element) data and solar flux indices ingested at 100Hz.' },
                      { title: 'Neo4j Graph Mapping', desc: 'Dynamic edge creation for satellites within 10km proximity threshold.' },
                      { title: 'TGNN Inference', desc: 'Temporal message passing to predict state at T+N minutes.' },
                      { title: 'Mission Control Alert', desc: 'Automated maneuver recommendation if probability > 1e-4.' },
                    ].map((step, i) => (
                      <div key={i} className="relative">
                        <div className="absolute -left-8 top-1 w-6 h-6 rounded-full bg-[#0a0a0c] border border-white/10 flex items-center justify-center text-[10px] font-bold text-blue-400">
                          {i + 1}
                        </div>
                        <h4 className="font-bold text-sm">{step.title}</h4>
                        <p className="text-xs text-white/40 mt-1">{step.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'vae' && (
            <motion.div
              key="vae"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              className="space-y-6"
            >
              <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
                <div className="flex flex-col xl:flex-row gap-8">
                  <div className="flex-1">
                    <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                      <Wind size={20} className="text-cyan-400" />
                      Turbine Telemetry VAE
                    </h3>
                    <p className="text-white/60 text-sm mb-6 leading-relaxed">
                      Variational Autoencoder used for predictive maintenance by detecting latent anomalies in turbine telemetry for atmospheric flight. It reconstructs input signals and flags deviations in latent space distribution.
                    </p>
                    <CodeBlock title="anomaly_vae.py" code={PYTHON_VAE_CODE} />
                  </div>
                  <div className="w-full xl:w-80 space-y-4">
                    <div className="p-4 bg-white/5 rounded-xl border border-white/5">
                      <h4 className="text-xs font-bold uppercase tracking-widest text-white/40 mb-4">Anomaly Indicators</h4>
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Reconstruction Loss</span>
                          <span className="text-green-400 font-mono text-sm">0.0024</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm">KL Divergence</span>
                          <span className="text-green-400 font-mono text-sm">1.12</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Latent Variance</span>
                          <span className="text-yellow-400 font-mono text-sm">0.85</span>
                        </div>
                      </div>
                    </div>
                    <div className="p-4 bg-red-500/10 rounded-xl border border-red-500/20">
                      <div className="flex items-center gap-2 text-red-400 mb-2">
                        <AlertTriangle size={14} />
                        <span className="text-xs font-bold uppercase">Latent Shift Detected</span>
                      </div>
                      <p className="text-[10px] text-white/60">
                        Turbine #4 showing 12% deviation in latent cluster centroid. Potential bearing degradation predicted in 45 flight hours.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'hardware' && (
            <motion.div
              key="hardware"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-6"
            >
              {HARDWARE_ANALYSIS.edge_processing.map((item, i) => (
                <div key={i} className="bg-white/5 border border-white/10 p-6 rounded-2xl flex gap-4">
                  <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center shrink-0">
                    <Cpu className="text-blue-400" size={24} />
                  </div>
                  <div>
                    <h4 className="text-white/40 text-[10px] uppercase tracking-widest font-bold mb-1">{item.component}</h4>
                    <div className="text-lg font-bold mb-2">{item.spec}</div>
                    <p className="text-sm text-white/60 leading-relaxed">{item.reason}</p>
                  </div>
                </div>
              ))}
              <div className="md:col-span-2 bg-blue-600/10 border border-blue-600/20 p-6 rounded-2xl">
                <h4 className="font-bold mb-2 flex items-center gap-2">
                  <Shield size={18} className="text-blue-400" />
                  Edge Deployment Strategy
                </h4>
                <p className="text-sm text-white/70 leading-relaxed">
                  Real-time edge processing is critical for low-latency collision avoidance. By deploying the TGNN model on NVIDIA Jetson Orin hardware within the satellite bus or atmospheric platform, we reduce the decision loop from minutes (ground-link) to milliseconds. This enables autonomous avoidance maneuvers without ground intervention in high-density orbital regimes.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Mobile Nav Overlay (Simple) */}
      <div className="lg:hidden fixed bottom-0 left-0 right-0 bg-[#0a0a0c] border-t border-white/10 p-2 flex justify-around z-50">
        {[
          { id: 'ssa', icon: Globe },
          { id: 'tgnn', icon: Layers },
          { id: 'vae', icon: Wind },
          { id: 'hardware', icon: Cpu },
        ].map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={`p-3 rounded-lg ${activeTab === item.id ? 'text-blue-400 bg-blue-400/10' : 'text-white/40'}`}
          >
            <item.icon size={20} />
          </button>
        ))}
      </div>
    </div>
  );
}
