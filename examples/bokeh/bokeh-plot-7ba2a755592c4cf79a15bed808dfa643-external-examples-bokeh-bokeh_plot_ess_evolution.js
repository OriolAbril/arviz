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
    
      
      
    
      var element = document.getElementById("7122ea0e-5b78-4bd3-a46d-c0b0c3751417");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '7122ea0e-5b78-4bd3-a46d-c0b0c3751417' but no matching script tag was found.")
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
                    
                  var docs_json = '{"e0576764-4325-4935-82a5-7e39a7e898d1":{"roots":{"references":[{"attributes":{},"id":"67967","type":"DataRange1d"},{"attributes":{},"id":"67976","type":"BasicTicker"},{"attributes":{},"id":"68040","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"68007"},"glyph":{"id":"68008"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68009"},"selection_glyph":null,"view":{"id":"68011"}},"id":"68010","type":"GlyphRenderer"},{"attributes":{"axis_label":"ESS","formatter":{"id":"68031"},"ticker":{"id":"67980"}},"id":"67979","type":"LinearAxis"},{"attributes":{},"id":"68038","type":"UnionRenderers"},{"attributes":{},"id":"67973","type":"LinearScale"},{"attributes":{"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"68013","type":"Line"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"68003","type":"Circle"},{"attributes":{"data_source":{"id":"68012"},"glyph":{"id":"68013"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68014"},"selection_glyph":null,"view":{"id":"68016"}},"id":"68015","type":"GlyphRenderer"},{"attributes":{"axis_label":"Total number of draws","formatter":{"id":"68029"},"ticker":{"id":"67976"}},"id":"67975","type":"LinearAxis"},{"attributes":{"data_source":{"id":"68017"},"glyph":{"id":"68018"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68019"},"selection_glyph":null,"view":{"id":"68021"}},"id":"68020","type":"GlyphRenderer"},{"attributes":{"label":{"value":"bulk"},"renderers":[{"id":"68005"},{"id":"68010"}]},"id":"68024","type":"LegendItem"},{"attributes":{"source":{"id":"68007"}},"id":"68011","type":"CDSView"},{"attributes":{},"id":"67984","type":"PanTool"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"68035"},"selection_policy":{"id":"68036"}},"id":"68007","type":"ColumnDataSource"},{"attributes":{"source":{"id":"68017"}},"id":"68021","type":"CDSView"},{"attributes":{"children":[{"id":"68046"},{"id":"68044"}]},"id":"68047","type":"Column"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"67992","type":"PolyAnnotation"},{"attributes":{},"id":"68035","type":"Selection"},{"attributes":{"click_policy":"hide","items":[{"id":"68024"},{"id":"68025"}],"location":"center_right","orientation":"horizontal"},"id":"68023","type":"Legend"},{"attributes":{"overlay":{"id":"67992"}},"id":"67987","type":"LassoSelectTool"},{"attributes":{},"id":"68036","type":"UnionRenderers"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"68037"},"selection_policy":{"id":"68038"}},"id":"68012","type":"ColumnDataSource"},{"attributes":{},"id":"68037","type":"Selection"},{"attributes":{},"id":"67983","type":"ResetTool"},{"attributes":{"children":[[{"id":"67966"},0,0]]},"id":"68044","type":"GridBox"},{"attributes":{"data_source":{"id":"68002"},"glyph":{"id":"68003"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68004"},"selection_glyph":null,"view":{"id":"68006"}},"id":"68005","type":"GlyphRenderer"},{"attributes":{},"id":"67988","type":"UndoTool"},{"attributes":{},"id":"67986","type":"WheelZoomTool"},{"attributes":{"text":"b"},"id":"68026","type":"Title"},{"attributes":{"toolbar":{"id":"68045"},"toolbar_location":"above"},"id":"68046","type":"ToolbarBox"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"68004","type":"Circle"},{"attributes":{"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"68008","type":"Line"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"67983"},{"id":"67984"},{"id":"67985"},{"id":"67986"},{"id":"67987"},{"id":"67988"},{"id":"67989"},{"id":"67990"}]},"id":"67993","type":"Toolbar"},{"attributes":{},"id":"68039","type":"Selection"},{"attributes":{},"id":"67969","type":"DataRange1d"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"68009","type":"Line"},{"attributes":{"source":{"id":"68002"}},"id":"68006","type":"CDSView"},{"attributes":{},"id":"67971","type":"LinearScale"},{"attributes":{"line_alpha":0.1,"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"68014","type":"Line"},{"attributes":{},"id":"68029","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"67991","type":"BoxAnnotation"},{"attributes":{"label":{"value":"tail"},"renderers":[{"id":"68020"},{"id":"68015"}]},"id":"68025","type":"LegendItem"},{"attributes":{"fill_color":{"value":"orange"},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"68018","type":"Circle"},{"attributes":{"callback":null},"id":"67990","type":"HoverTool"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"68039"},"selection_policy":{"id":"68040"}},"id":"68017","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"orange"},"line_alpha":{"value":0.1},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"68019","type":"Circle"},{"attributes":{"axis":{"id":"67975"},"ticker":null},"id":"67978","type":"Grid"},{"attributes":{"axis":{"id":"67979"},"dimension":1,"ticker":null},"id":"67982","type":"Grid"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"68022","type":"Span"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"68033"},"selection_policy":{"id":"68034"}},"id":"68002","type":"ColumnDataSource"},{"attributes":{},"id":"68031","type":"BasicTickFormatter"},{"attributes":{},"id":"68033","type":"Selection"},{"attributes":{"source":{"id":"68012"}},"id":"68016","type":"CDSView"},{"attributes":{"overlay":{"id":"67991"}},"id":"67985","type":"BoxZoomTool"},{"attributes":{},"id":"68034","type":"UnionRenderers"},{"attributes":{},"id":"67980","type":"BasicTicker"},{"attributes":{"toolbars":[{"id":"67993"}],"tools":[{"id":"67983"},{"id":"67984"},{"id":"67985"},{"id":"67986"},{"id":"67987"},{"id":"67988"},{"id":"67989"},{"id":"67990"}]},"id":"68045","type":"ProxyToolbar"},{"attributes":{"above":[{"id":"68023"}],"below":[{"id":"67975"}],"center":[{"id":"67978"},{"id":"67982"}],"left":[{"id":"67979"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"68005"},{"id":"68010"},{"id":"68015"},{"id":"68020"},{"id":"68022"}],"title":{"id":"68026"},"toolbar":{"id":"67993"},"toolbar_location":null,"x_range":{"id":"67967"},"x_scale":{"id":"67971"},"y_range":{"id":"67969"},"y_scale":{"id":"67973"}},"id":"67966","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"67989","type":"SaveTool"}],"root_ids":["68047"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"e0576764-4325-4935-82a5-7e39a7e898d1","root_ids":["68047"],"roots":{"68047":"7122ea0e-5b78-4bd3-a46d-c0b0c3751417"}}];
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