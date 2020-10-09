(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("731d2719-6ca9-459e-99f9-a81a5ff7bf9f");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '731d2719-6ca9-459e-99f9-a81a5ff7bf9f' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"1955f40a-11c4-4c47-a167-ac3e74af8c90":{"roots":{"references":[{"attributes":{"overlay":{"id":"65566"}},"id":"65561","type":"LassoSelectTool"},{"attributes":{"source":{"id":"65588"}},"id":"65592","type":"CDSView"},{"attributes":{},"id":"65560","type":"WheelZoomTool"},{"attributes":{},"id":"65562","type":"UndoTool"},{"attributes":{"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"65584","type":"MultiLine"},{"attributes":{"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"65594","type":"MultiLine"},{"attributes":{},"id":"65612","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"65566","type":"PolyAnnotation"},{"attributes":{"below":[{"id":"65549"}],"center":[{"id":"65552"},{"id":"65556"}],"left":[{"id":"65553"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"65581"},{"id":"65586"},{"id":"65591"},{"id":"65596"},{"id":"65601"},{"id":"65603"}],"title":{"id":"65604"},"toolbar":{"id":"65567"},"toolbar_location":null,"x_range":{"id":"65541"},"x_scale":{"id":"65545"},"y_range":{"id":"65543"},"y_scale":{"id":"65547"}},"id":"65540","subtype":"Figure","type":"Plot"},{"attributes":{"end":0.5,"start":-1.5},"id":"65543","type":"DataRange1d"},{"attributes":{"fill_color":{"value":"black"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"65599","type":"Circle"},{"attributes":{"data":{"x":[-32.37106695144684,-32.71848009989285],"y":[0.0,-1.0]},"selected":{"id":"65617"},"selection_policy":{"id":"65618"}},"id":"65598","type":"ColumnDataSource"},{"attributes":{},"id":"65614","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"grey"},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"65579","type":"Triangle"},{"attributes":{"children":[{"id":"65624"},{"id":"65622"}]},"id":"65625","type":"Column"},{"attributes":{},"id":"65541","type":"DataRange1d"},{"attributes":{},"id":"65611","type":"Selection"},{"attributes":{},"id":"65563","type":"SaveTool"},{"attributes":{},"id":"65609","type":"Selection"},{"attributes":{"children":[[{"id":"65540"},0,0]]},"id":"65622","type":"GridBox"},{"attributes":{},"id":"65617","type":"Selection"},{"attributes":{"axis":{"id":"65553"},"dimension":1,"ticker":null},"id":"65556","type":"Grid"},{"attributes":{},"id":"65606","type":"BasicTickFormatter"},{"attributes":{},"id":"65615","type":"Selection"},{"attributes":{},"id":"65608","type":"BasicTickFormatter"},{"attributes":{},"id":"65616","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":null},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"65590","type":"Circle"},{"attributes":{"source":{"id":"65598"}},"id":"65602","type":"CDSView"},{"attributes":{"formatter":{"id":"65608"},"major_label_overrides":{"-0.75":"","-1":"Centered 8 schools","0":"Non-centered 8 schools"},"ticker":{"id":"65576"}},"id":"65553","type":"LinearAxis"},{"attributes":{"line_alpha":{"value":0.1},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"65595","type":"MultiLine"},{"attributes":{"fill_color":{"value":null},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"65589","type":"Circle"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"xs":{"field":"xs"},"ys":{"field":"ys"}},"id":"65585","type":"MultiLine"},{"attributes":{"axis_label":"Log","formatter":{"id":"65606"},"ticker":{"id":"65550"}},"id":"65549","type":"LinearAxis"},{"attributes":{"source":{"id":"65593"}},"id":"65597","type":"CDSView"},{"attributes":{"ticks":[0.0,-0.75,-1.0]},"id":"65576","type":"FixedTicker"},{"attributes":{"text":""},"id":"65604","type":"Title"},{"attributes":{"data_source":{"id":"65588"},"glyph":{"id":"65589"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"65590"},"selection_glyph":null,"view":{"id":"65592"}},"id":"65591","type":"GlyphRenderer"},{"attributes":{},"id":"65610","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"grey"},"line_alpha":{"value":0.1},"line_color":{"value":"grey"},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"65580","type":"Triangle"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"65557"},{"id":"65558"},{"id":"65559"},{"id":"65560"},{"id":"65561"},{"id":"65562"},{"id":"65563"},{"id":"65564"}]},"id":"65567","type":"Toolbar"},{"attributes":{},"id":"65613","type":"Selection"},{"attributes":{},"id":"65618","type":"UnionRenderers"},{"attributes":{"toolbars":[{"id":"65567"}],"tools":[{"id":"65557"},{"id":"65558"},{"id":"65559"},{"id":"65560"},{"id":"65561"},{"id":"65562"},{"id":"65563"},{"id":"65564"}]},"id":"65623","type":"ProxyToolbar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"65565","type":"BoxAnnotation"},{"attributes":{"callback":null},"id":"65564","type":"HoverTool"},{"attributes":{},"id":"65547","type":"LinearScale"},{"attributes":{"data_source":{"id":"65593"},"glyph":{"id":"65594"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"65595"},"selection_glyph":null,"view":{"id":"65597"}},"id":"65596","type":"GlyphRenderer"},{"attributes":{"data":{"x":[-30.687290318389813,-30.81037417660005],"y":[0.0,-1.0]},"selected":{"id":"65613"},"selection_policy":{"id":"65614"}},"id":"65588","type":"ColumnDataSource"},{"attributes":{"data":{"x":[-30.81037417660005],"y":[-0.75]},"selected":{"id":"65609"},"selection_policy":{"id":"65610"}},"id":"65578","type":"ColumnDataSource"},{"attributes":{"toolbar":{"id":"65623"},"toolbar_location":"above"},"id":"65624","type":"ToolbarBox"},{"attributes":{},"id":"65545","type":"LinearScale"},{"attributes":{},"id":"65557","type":"ResetTool"},{"attributes":{"data_source":{"id":"65583"},"glyph":{"id":"65584"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"65585"},"selection_glyph":null,"view":{"id":"65587"}},"id":"65586","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"65598"},"glyph":{"id":"65599"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"65600"},"selection_glyph":null,"view":{"id":"65602"}},"id":"65601","type":"GlyphRenderer"},{"attributes":{},"id":"65558","type":"PanTool"},{"attributes":{"data":{"xs":[[-30.896420573800537,-30.724327779399562]],"ys":[[-0.75,-0.75]]},"selected":{"id":"65611"},"selection_policy":{"id":"65612"}},"id":"65583","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"black"},"line_alpha":{"value":0.1},"line_width":{"value":2},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"65600","type":"Circle"},{"attributes":{"axis":{"id":"65549"},"ticker":null},"id":"65552","type":"Grid"},{"attributes":{"dimension":"height","line_color":"grey","line_dash":[6],"line_width":1.7677669529663689,"location":-30.687290318389813},"id":"65603","type":"Span"},{"attributes":{"source":{"id":"65578"}},"id":"65582","type":"CDSView"},{"attributes":{"data":{"xs":[[-31.9932025950496,-29.381378041730027],[-32.060105903049745,-29.56064245015035]],"ys":[[0.0,0.0],[-1.0,-1.0]]},"selected":{"id":"65615"},"selection_policy":{"id":"65616"}},"id":"65593","type":"ColumnDataSource"},{"attributes":{},"id":"65550","type":"BasicTicker"},{"attributes":{"data_source":{"id":"65578"},"glyph":{"id":"65579"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"65580"},"selection_glyph":null,"view":{"id":"65582"}},"id":"65581","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"65565"}},"id":"65559","type":"BoxZoomTool"},{"attributes":{"source":{"id":"65583"}},"id":"65587","type":"CDSView"}],"root_ids":["65625"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"1955f40a-11c4-4c47-a167-ac3e74af8c90","root_ids":["65625"],"roots":{"65625":"731d2719-6ca9-459e-99f9-a81a5ff7bf9f"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();