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
    
      
      
    
      var element = document.getElementById("93e1d9b6-f04c-4075-89a6-27d5f5bb105c");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '93e1d9b6-f04c-4075-89a6-27d5f5bb105c' but no matching script tag was found.")
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
                    
                  var docs_json = '{"61d603fa-8c51-4682-93d3-b9fe954d189b":{"roots":{"references":[{"attributes":{},"id":"4178","type":"UnionRenderers"},{"attributes":{"axis_label":"Total number of draws","formatter":{"id":"4167"},"ticker":{"id":"4114"}},"id":"4113","type":"LinearAxis"},{"attributes":{"children":[{"id":"4184"},{"id":"4182"}]},"id":"4185","type":"Column"},{"attributes":{"line_alpha":0.1,"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"4152","type":"Line"},{"attributes":{"axis_label":"ESS","formatter":{"id":"4169"},"ticker":{"id":"4118"}},"id":"4117","type":"LinearAxis"},{"attributes":{"source":{"id":"4150"}},"id":"4154","type":"CDSView"},{"attributes":{"callback":null},"id":"4128","type":"HoverTool"},{"attributes":{},"id":"4107","type":"DataRange1d"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"orange"},"line_alpha":{"value":0.1},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4157","type":"Circle"},{"attributes":{},"id":"4114","type":"BasicTicker"},{"attributes":{"axis":{"id":"4113"},"ticker":null},"id":"4116","type":"Grid"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4141","type":"Circle"},{"attributes":{"line_alpha":0.1,"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"4147","type":"Line"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4179"},"selection_policy":{"id":"4180"}},"id":"4155","type":"ColumnDataSource"},{"attributes":{"source":{"id":"4155"}},"id":"4159","type":"CDSView"},{"attributes":{"fill_color":{"value":"orange"},"line_color":{"value":"orange"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4156","type":"Circle"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4130","type":"PolyAnnotation"},{"attributes":{},"id":"4179","type":"Selection"},{"attributes":{"axis":{"id":"4117"},"dimension":1,"ticker":null},"id":"4120","type":"Grid"},{"attributes":{"data_source":{"id":"4145"},"glyph":{"id":"4146"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4147"},"selection_glyph":null,"view":{"id":"4149"}},"id":"4148","type":"GlyphRenderer"},{"attributes":{"label":{"value":"bulk"},"renderers":[{"id":"4143"},{"id":"4148"}]},"id":"4162","type":"LegendItem"},{"attributes":{},"id":"4118","type":"BasicTicker"},{"attributes":{},"id":"4180","type":"UnionRenderers"},{"attributes":{"text":"b"},"id":"4164","type":"Title"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4142","type":"Circle"},{"attributes":{"overlay":{"id":"4129"}},"id":"4123","type":"BoxZoomTool"},{"attributes":{"data_source":{"id":"4155"},"glyph":{"id":"4156"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4157"},"selection_glyph":null,"view":{"id":"4159"}},"id":"4158","type":"GlyphRenderer"},{"attributes":{},"id":"4122","type":"PanTool"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4173"},"selection_policy":{"id":"4174"}},"id":"4140","type":"ColumnDataSource"},{"attributes":{},"id":"4121","type":"ResetTool"},{"attributes":{"children":[[{"id":"4104"},0,0]]},"id":"4182","type":"GridBox"},{"attributes":{},"id":"4105","type":"DataRange1d"},{"attributes":{},"id":"4127","type":"SaveTool"},{"attributes":{"click_policy":"hide","items":[{"id":"4162"},{"id":"4163"}],"location":"center_right","orientation":"horizontal"},"id":"4161","type":"Legend"},{"attributes":{"label":{"value":"tail"},"renderers":[{"id":"4158"},{"id":"4153"}]},"id":"4163","type":"LegendItem"},{"attributes":{},"id":"4124","type":"WheelZoomTool"},{"attributes":{"toolbar":{"id":"4183"},"toolbar_location":"above"},"id":"4184","type":"ToolbarBox"},{"attributes":{"above":[{"id":"4161"}],"below":[{"id":"4113"}],"center":[{"id":"4116"},{"id":"4120"}],"left":[{"id":"4117"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"4143"},{"id":"4148"},{"id":"4153"},{"id":"4158"},{"id":"4160"}],"title":{"id":"4164"},"toolbar":{"id":"4131"},"toolbar_location":null,"x_range":{"id":"4105"},"x_scale":{"id":"4109"},"y_range":{"id":"4107"},"y_scale":{"id":"4111"}},"id":"4104","subtype":"Figure","type":"Plot"},{"attributes":{"overlay":{"id":"4130"}},"id":"4125","type":"LassoSelectTool"},{"attributes":{},"id":"4126","type":"UndoTool"},{"attributes":{},"id":"4169","type":"BasicTickFormatter"},{"attributes":{},"id":"4173","type":"Selection"},{"attributes":{"toolbars":[{"id":"4131"}],"tools":[{"id":"4121"},{"id":"4122"},{"id":"4123"},{"id":"4124"},{"id":"4125"},{"id":"4126"},{"id":"4127"},{"id":"4128"}]},"id":"4183","type":"ProxyToolbar"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4121"},{"id":"4122"},{"id":"4123"},{"id":"4124"},{"id":"4125"},{"id":"4126"},{"id":"4127"},{"id":"4128"}]},"id":"4131","type":"Toolbar"},{"attributes":{},"id":"4175","type":"Selection"},{"attributes":{},"id":"4174","type":"UnionRenderers"},{"attributes":{"source":{"id":"4140"}},"id":"4144","type":"CDSView"},{"attributes":{},"id":"4176","type":"UnionRenderers"},{"attributes":{"line_color":"#1f77b4","x":{"field":"x"},"y":{"field":"y"}},"id":"4146","type":"Line"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"0glo/ZHQYUCWuV3JG5ppQMqT1P2iwXVAqY8DrPB3e0Bh+Z3qJ/WAQBkscwHQK4RAmBvOlEfWikBL3Hu21ASOQLfWD/YkmpBAzy5bwfaykkBL+YYwKf6TQNX3Ngx5b5VAciIDv7ZVlkBpY4usN1KYQKnVB/o5gplAPOgo6bOym0CHt7M3Mq+fQCfqDPZjtqBAJ784mgSeoUD/m8TvrdqhQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4175"},"selection_policy":{"id":"4176"}},"id":"4145","type":"ColumnDataSource"},{"attributes":{},"id":"4109","type":"LinearScale"},{"attributes":{"data_source":{"id":"4140"},"glyph":{"id":"4141"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4142"},"selection_glyph":null,"view":{"id":"4144"}},"id":"4143","type":"GlyphRenderer"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4129","type":"BoxAnnotation"},{"attributes":{},"id":"4167","type":"BasicTickFormatter"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAWUAAAAAAAABpQAAAAAAAwHJAAAAAAAAAeUAAAAAAAEB/QAAAAAAAwIJAAAAAAADghUAAAAAAAACJQAAAAAAAIIxAAAAAAABAj0AAAAAAADCRQAAAAAAAwJJAAAAAAABQlEAAAAAAAOCVQAAAAAAAcJdAAAAAAAAAmUAAAAAAAJCaQAAAAAAAIJxAAAAAAACwnUAAAAAAAECfQA==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"kUgpITo5UkD/4xVCu9ZkQIwda8wmdG9AKVV7klfmc0B9iLN6f4t2QCN384Oug3lAXRIP57k5gEDe8oZZBOGEQCI3trf/S4dAdSETuLRMikCa5sM7ewSJQLpDDoqDVoxADogAvE7ZjUBXvDQTTveQQPztYWbX55BATSx1OrcnkUCJGFkHolCTQIz00+lP8ZNAUA/0blR0lEAOVbAHnSuWQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4177"},"selection_policy":{"id":"4178"}},"id":"4150","type":"ColumnDataSource"},{"attributes":{"line_color":"orange","x":{"field":"x"},"y":{"field":"y"}},"id":"4151","type":"Line"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"4160","type":"Span"},{"attributes":{},"id":"4177","type":"Selection"},{"attributes":{"data_source":{"id":"4150"},"glyph":{"id":"4151"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4152"},"selection_glyph":null,"view":{"id":"4154"}},"id":"4153","type":"GlyphRenderer"},{"attributes":{},"id":"4111","type":"LinearScale"},{"attributes":{"source":{"id":"4145"}},"id":"4149","type":"CDSView"}],"root_ids":["4185"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"61d603fa-8c51-4682-93d3-b9fe954d189b","root_ids":["4185"],"roots":{"4185":"93e1d9b6-f04c-4075-89a6-27d5f5bb105c"}}];
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