/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { 
  Upload, 
  Microscope, 
  Search, 
  Waves, 
  Layers, 
  Droplets,
  ChevronDown,
  Info,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Eye,
  Thermometer,
  ShieldAlert
} from "lucide-react";
import { cn } from "@/src/lib/utils";

// --- Types ---

interface EnvironmentalMetrics {
  temperature: string;
  dhw: string;
  ssta: string;
  turbidity: string;
  windspeed: string;
  sheltered: string;
}

interface AnalysisResult {
  prediction: string;         // "healthy" | "bleached"
  risk: string;               // "Low" | "Moderate" | "High"
  finalProb: number;          // 0-100 (final fused probability %)
  cnnProb: number;            // 0-100 (image model confidence %)
  xgbProb: number | null;     // 0-100 (env model score %) or null
  mode: string;               // "multimodal" | "image_only"
  summary: string;
  recommendations: string[];
}

// --- Helpers ---

const getRiskColor = (risk: string) => {
  if (risk === "High") return "text-error";
  if (risk === "Moderate") return "text-secondary";
  return "text-primary";
};

// --- Components ---

const Header = () => (
  <header className="fixed top-0 w-full z-50 bg-surface/80 backdrop-blur-xl border-b border-outline-variant/20">
    <div className="w-full px-8 py-4 flex justify-between items-center">
      <div className="text-xl font-bold tracking-tight text-primary flex items-center gap-2">
        <Droplets className="w-6 h-6" />
        DeepReef 
      </div>

    </div>
  </header>
);

const Hero = () => (
  <section className="mb-16">
    <motion.h1 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="text-6xl font-extrabold tracking-tight text-primary mb-4 leading-[1.1]"
    >
      DeepReef: Coral Health Analysis ~
    </motion.h1>
    <motion.p 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="text-on-surface-variant text-xl max-w-2xl leading-relaxed"
    >
      Exploring how coral reefs respond to changing ocean conditions, revealing the early signs of stress and resilience beneath the surface.
    </motion.p>
  </section>
);

const Footer = () => (
  <footer className="w-full py-5 mt-10 bg-surface-container-low border-t border-outline-variant/20">
    <div className="w-full px-8 flex flex-col md:flex-row justify-between items-center gap-6">
      <div className="font-bold text-on-surface flex items-center gap-2">
        <Droplets className="w-5 h-5 text-primary" />
        DeepReef 
      </div>
      <div className="flex gap-8 text-[0.75rem] tracking-wide font-medium">
        <a className="text-on-surface-variant opacity-80 hover:opacity-100 hover:text-secondary transition-all underline underline-offset-4" href="#"></a>

      </div>
      <p className="text-on-surface-variant text-[0.75rem] opacity-80">
        Predictions are generated using machine learning models and may be subject to uncertainty due to environmental variability and data limitations.
      </p>
    </div>
  </footer>
);

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Updated defaults — realistic reef site values
  const [metrics, setMetrics] = useState<EnvironmentalMetrics>({
    temperature: "27.5",
    dhw: "2",
    ssta: "0.4",
    turbidity: "0.05",
    windspeed: "6",
    sheltered: "Yes"
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!image || !fileInputRef.current?.files?.[0]) return;

    setIsAnalyzing(true);
    setResult(null);
    setError(null);

    try {
      const file = fileInputRef.current.files[0];

      const formData = new FormData();
      formData.append("image", file);
      formData.append("temperature", metrics.temperature);
      formData.append("dhw", metrics.dhw);
      formData.append("ssta", metrics.ssta);
      formData.append("turbidity", metrics.turbidity);
      formData.append("windspeed", metrics.windspeed);
      formData.append("sheltered", metrics.sheltered === "Yes" ? "1" : "0");

      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Backend error");

      const data = await res.json();

      // Map backend response to frontend — all real values, nothing fabricated
      setResult({
        prediction: data.prediction,
        risk:       data.risk,
        finalProb:  Math.round(data.final_prob * 100),
        cnnProb:    Math.round(data.cnn_prob * 100),
        xgbProb:    data.xgb_prob !== null ? Math.round(data.xgb_prob * 100) : null,
        mode:       data.mode,
        summary:    data.explanation,
        recommendations:
          data.risk === "High"
            ? [
                "Reduce thermal stress exposure",
                "Increase monitoring frequency",
                "Report to reef authorities",
                "Consider intervention measures",
              ]
            : data.risk === "Moderate"
            ? [
                "Monitor reef regularly",
                "Track environmental changes",
                "Maintain observation logs",
              ]
            : [
                "Maintain current conditions",
                "Continue periodic monitoring",
                "Record baseline health data",
              ],
      });
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 pt-22 pb-24 px-8 w-full max-w-7xl mx-auto">
        <Hero />

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left Column: Input Panel */}
          <div className="lg:col-span-4 space-y-8">
            {/* Image Upload */}
            <div className="bg-surface-container-low p-8 rounded-xl border border-outline-variant/10 shadow-sm">
              <h2 className="text-sm font-bold uppercase tracking-widest text-primary mb-6 flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-primary text-on-primary flex items-center justify-center text-[10px]">1</span>
                Coral Image
              </h2>
              <div 
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  "relative group cursor-pointer border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center transition-all duration-300",
                  image ? "border-primary/50 bg-primary/5" : "border-outline-variant/30 bg-surface-container-lowest hover:bg-surface-container"
                )}
              >
                {image ? (
                  <div className="relative w-full aspect-square rounded-md overflow-hidden shadow-inner">
                    <img src={image} alt="Preview" className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <Upload className="text-white w-8 h-8" />
                    </div>
                  </div>
                ) : (
                  <>
                    <Upload className="w-10 h-10 text-primary mb-4 group-hover:scale-110 transition-transform" />
                    <span className="text-on-surface font-semibold text-center">Drop Reef Imagery Here</span>
                    <span className="text-on-surface-variant text-xs mt-2 text-center">PNG or JPEG (Max 50MB)</span>
                  </>
                )}
                <input 
                  ref={fileInputRef}
                  className="hidden" 
                  type="file" 
                  accept="image/*"
                  onChange={handleImageUpload}
                />
              </div>
            </div>

            {/* Environmental Data */}
            <div className="bg-surface-container-low p-8 rounded-xl border border-outline-variant/10 shadow-sm">
              <h2 className="text-sm font-bold uppercase tracking-widest text-primary mb-6 flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-primary text-on-primary flex items-center justify-center text-[10px]">2</span>
                Environmental Metrics
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {[
                  { label: "Temperature (°C)", key: "temperature", placeholder: "27.5" },
                  { label: "DHW (weeks)", key: "dhw", placeholder: "2" },
                  { label: "SSTA (°C)", key: "ssta", placeholder: "0.4" },
                  { label: "Turbidity (0–1)", key: "turbidity", placeholder: "0.05" },
                  { label: "Windspeed (knots)", key: "windspeed", placeholder: "6" },
                ].map((input) => (
                  <div key={input.key} className="flex flex-col gap-2">
                    <label className="text-xs font-semibold text-on-surface-variant">{input.label}</label>
                    <input 
                      type="number"
                      step="0.1"
                      className="bg-surface-container-high border-none rounded-md p-3 focus:ring-2 focus:ring-primary text-on-surface transition-all outline-none"
                      placeholder={input.placeholder}
                      value={metrics[input.key as keyof EnvironmentalMetrics]}
                      onChange={(e) => setMetrics({ ...metrics, [input.key]: e.target.value })}
                    />
                  </div>
                ))}
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-semibold text-on-surface-variant">Sheltered site</label>
                  <div className="relative">
                    <select 
                      className="w-full bg-surface-container-high border-none rounded-md p-3 focus:ring-2 focus:ring-primary text-on-surface appearance-none outline-none cursor-pointer"
                      value={metrics.sheltered}
                      onChange={(e) => setMetrics({ ...metrics, sheltered: e.target.value })}
                    >
                      <option>Yes</option>
                      <option>No</option>
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-on-surface-variant pointer-events-none" />
                  </div>
                </div>
              </div>
            </div>

            {/* Action Button */}
            <button 
              onClick={handleAnalyze}
              disabled={!image || isAnalyzing}
              className={cn(
                "w-full py-5 rounded-xl font-bold text-lg flex items-center justify-center gap-3 transition-all shadow-lg active:scale-[0.98]",
                !image || isAnalyzing 
                  ? "bg-surface-container-highest text-on-surface-variant cursor-not-allowed" 
                  : "bg-primary text-on-primary hover:bg-primary/90 hover:shadow-primary/20"
              )}
            >
              {isAnalyzing ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                >
                  <Activity className="w-6 h-6" />
                </motion.div>
              ) : (
                <Microscope className="w-6 h-6" />
              )}
              {isAnalyzing ? "Processing..." : "Analyse Coral"}
            </button>
          </div>

          {/* Right Column: Results Display */}
          <div className="lg:col-span-8">
            <div className="bg-surface-container p-1 rounded-xl h-full flex flex-col overflow-hidden min-h-[600px] border border-outline-variant/10 shadow-sm">
              <div className="flex-1 flex flex-col bg-surface-container-low rounded-lg p-8 relative overflow-y-auto">
                <AnimatePresence mode="wait">
                  {!result && !isAnalyzing && (
                    <motion.div 
                      key="placeholder"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex-1 flex flex-col items-center justify-center text-center"
                    >
                      <div className="w-full max-w-2xl bg-surface-container-lowest/50 rounded-2xl overflow-hidden aspect-video relative flex items-center justify-center border border-outline-variant/5">
                        <div className="absolute inset-0 opacity-10 bg-primary/20 animate-pulse"></div>
                        <Search className="w-16 h-16 text-outline-variant/50" />
                      </div>
                      <h3 className="mt-8 text-2xl font-bold text-primary">Awaiting Analysis</h3>
                      <p className="text-on-surface-variant mt-2 max-w-sm">
                        Upload a coral reef image and provide environmental metrics to generate a health diagnostic report.
                      </p>
                      
                      {/* Placeholder cards — greyed out */}
                      <div className="mt-12 w-full grid grid-cols-1 sm:grid-cols-3 gap-4 opacity-30 grayscale pointer-events-none">
                        {[
                          { label: "Risk level", value: "--" },
                          { label: "Image signal", value: "--%"},
                          { label: "Env. stress", value: "--%"},
                        ].map((stat, i) => (
                          <div key={i} className="bg-surface-container-highest p-4 rounded-lg text-left">
                            <div className="text-[10px] uppercase font-bold text-on-surface-variant">{stat.label}</div>
                            <div className="text-2xl font-bold text-primary">{stat.value}</div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}

                  {isAnalyzing && (
                    <motion.div 
                      key="analyzing"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex-1 flex flex-col items-center justify-center text-center space-y-6"
                    >
                      <div className="relative">
                        <motion.div 
                          animate={{ scale: [1, 1.1, 1], opacity: [0.5, 1, 0.5] }}
                          transition={{ repeat: Infinity, duration: 2 }}
                          className="absolute inset-0 bg-primary/20 rounded-full blur-2xl"
                        />
                        <Activity className="w-20 h-20 text-primary relative z-10" />
                      </div>
                      <div className="space-y-2">
                        <h3 className="text-2xl font-bold text-primary">Analyzing Reef...</h3>
                        <p className="text-on-surface-variant max-w-xs mx-auto">
                          Running image classification and cross-referencing environmental data.
                        </p>
                      </div>
                      <div className="w-48 h-1 bg-surface-container-highest rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ x: "-100%" }}
                          animate={{ x: "100%" }}
                          transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
                          className="w-1/2 h-full bg-primary"
                        />
                      </div>
                    </motion.div>
                  )}

                  {error && (
                    <motion.div 
                      key="error"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex-1 flex flex-col items-center justify-center text-center space-y-4"
                    >
                      <AlertTriangle className="w-16 h-16 text-error" />
                      <div className="space-y-2">
                        <h3 className="text-2xl font-bold text-error">Analysis Error</h3>
                        <p className="text-on-surface-variant max-w-xs mx-auto">{error}</p>
                      </div>
                      <button 
                        onClick={() => setError(null)}
                        className="px-6 py-2 rounded-lg bg-surface-container-highest text-on-surface font-semibold hover:bg-outline-variant/20 transition-colors"
                      >
                        Try Again
                      </button>
                    </motion.div>
                  )}

                  {result && (
                    <motion.div 
                      key="result"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-8"
                    >
                      {/* Header row */}
                      <div className="flex items-center justify-between flex-wrap gap-4">
                        <div>
                          <h3 className="text-3xl font-bold text-primary">Analysis Report</h3>
                          <p className="text-xs text-on-surface-variant mt-1 uppercase tracking-wider font-semibold">
                            Mode: {result.mode === "multimodal" ? "Image + Environmental data" : "Image only"}
                          </p>
                        </div>
                        <div className="px-4 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4" />
                          Diagnostic complete
                        </div>
                      </div>

                      {/* ── Real data stat cards ── */}
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        {/* Card 1: Risk level */}
                        <div className="bg-surface-container-highest p-6 rounded-xl border border-outline-variant/10 flex flex-col gap-2">
                          <div className="flex items-center gap-2 text-[10px] uppercase font-bold text-on-surface-variant">
                            <ShieldAlert className="w-3.5 h-3.5" />
                            Risk level
                          </div>
                          <div className={cn("text-3xl font-black", getRiskColor(result.risk))}>
                            {result.risk}
                          </div>
                          <div className="text-xs text-on-surface-variant">
                            {result.prediction === "bleached" ? "Bleaching detected" : "No bleaching detected"}
                          </div>
                        </div>

                        {/* Card 2: Image model signal */}
                        <div className="bg-surface-container-highest p-6 rounded-xl border border-outline-variant/10 flex flex-col gap-2">
                          <div className="flex items-center gap-2 text-[10px] uppercase font-bold text-on-surface-variant">
                            <Eye className="w-3.5 h-3.5" />
                            Image signal
                          </div>
                          <div className="text-3xl font-black text-primary">
                            {result.cnnProb}%
                          </div>
                          <div className="text-xs text-on-surface-variant">
                            CNN bleaching probability
                          </div>
                        </div>

                        {/* Card 3: Environmental stress or fused score */}
                        <div className="bg-surface-container-highest p-6 rounded-xl border border-outline-variant/10 flex flex-col gap-2">
                          <div className="flex items-center gap-2 text-[10px] uppercase font-bold text-on-surface-variant">
                            <Thermometer className="w-3.5 h-3.5" />
                            {result.xgbProb !== null ? "Env. stress" : "Final score"}
                          </div>
                          <div className="text-3xl font-black text-secondary">
                            {result.xgbProb !== null ? `${result.xgbProb}%` : `${result.finalProb}%`}
                          </div>
                          <div className="text-xs text-on-surface-variant">
                            {result.xgbProb !== null
                              ? "Environmental model score"
                              : "Fused confidence score"}
                          </div>
                        </div>
                      </div>

                      {/* Summary + Recommendations */}
                      <div className="bg-surface-container-lowest p-8 rounded-2xl border border-outline-variant/10 space-y-6">
                        <div className="space-y-2">
                          <h4 className="text-sm font-bold uppercase tracking-widest text-on-surface-variant flex items-center gap-2">
                            <Info className="w-4 h-4" />
                            Assessment summary
                          </h4>
                          <p className="text-on-surface leading-relaxed text-lg italic font-medium">
                            "{result.summary}"
                          </p>
                        </div>

                        <div className="h-px bg-outline-variant/20" />

                        <div className="space-y-4">
                          <h4 className="text-sm font-bold uppercase tracking-widest text-on-surface-variant flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-secondary" />
                            Recommendations
                          </h4>
                          <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {result.recommendations.map((rec, i) => (
                              <li key={i} className="flex items-start gap-3 p-3 rounded-lg bg-surface-container-low border border-outline-variant/5 text-sm text-on-surface">
                                <div className="mt-1 w-1.5 h-1.5 rounded-full bg-secondary shrink-0" />
                                {rec}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex justify-end gap-4">
                        <button 
                          onClick={() => setResult(null)}
                          className="px-6 py-2 rounded-lg bg-primary text-on-primary font-semibold hover:opacity-90 transition-opacity shadow-md"
                        >
                          Reset
                        </button>
      
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </div>

        {/* Methodology Section */}
        <section className="mt-16 grid md:grid-cols-2 gap-16 items-center">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="order-2 md:order-1"
          >
            <div className="rounded-3xl overflow-hidden shadow-2xl aspect-square bg-surface-container relative group">
              <img 
                alt="Scientific Coral Observation" 
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" 
                src="/coral.jpeg"
                referrerPolicy="no-referrer"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-primary/40 to-transparent opacity-60" />
            </div>
          </motion.div>
          
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="order-1 md:order-2 space-y-8"
          >
            <span className="inline-block px-4 py-1.5 rounded-full bg-tertiary-container text-on-tertiary-fixed text-xs font-bold tracking-widest uppercase shadow-sm">
              Understanding Corals
            </span>
            <h2 className="text-5xl font-bold text-primary leading-[1.15]">
            Corals ~ Ocean’s Living Architecture
            </h2>
            <p className="text-on-surface-variant leading-relaxed text-lg">
     Beneath the surface, coral reefs exist in a delicate equilibrium, quietly responding to shifts in temperature, light, and ocean currents. Built over decades by countless tiny organisms working in unison, they form vast, intricate structures that support entire marine ecosystems.
  
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="p-8 bg-surface-container-low rounded-2xl border border-outline-variant/10 flex flex-col gap-4 hover:shadow-md transition-shadow">
                <Waves className="w-8 h-8 text-primary" />
                <div>
                  <h4 className="font-bold text-primary">Ocean Keystone</h4>
                  <p className="text-xs text-on-surface-variant mt-1">Coral reefs support over 25% of all marine species and act as critical buffers that protect coastlines from erosion and storm damage.</p>
                </div>
              </div>
              <div className="p-8 bg-surface-container-low rounded-2xl border border-outline-variant/10 flex flex-col gap-4 hover:shadow-md transition-shadow">
                <Layers className="w-8 h-8 text-secondary" />
                <div>
                  <h4 className="font-bold text-secondary">Thermal Stress</h4>
                  <p className="text-xs text-on-surface-variant mt-1">  
  When ocean temperatures rise, corals lose the algae that keep them alive, leading to bleaching and, if prolonged, the collapse of entire reef systems.</p>
                </div>
              </div>
            </div>
          </motion.div>
        </section>
      </main>

      <Footer />
    </div>
  );
}