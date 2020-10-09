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
    
      
      
    
      var element = document.getElementById("ab4244ec-d320-4b71-9ba7-d9983f1b1ca6");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'ab4244ec-d320-4b71-9ba7-d9983f1b1ca6' but no matching script tag was found.")
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
                    
                  var docs_json = '{"52e1ec3a-f208-440e-9d9d-023e6b226d4e":{"roots":{"references":[{"attributes":{"data_source":{"id":"68150"},"glyph":{"id":"68149"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"68152"}},"id":"68151","type":"GlyphRenderer"},{"attributes":{},"id":"68122","type":"BasicTicker"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"68134","type":"PolyAnnotation"},{"attributes":{},"id":"68165","type":"UnionRenderers"},{"attributes":{},"id":"68125","type":"ResetTool"},{"attributes":{"data_source":{"id":"68144"},"glyph":{"id":"68145"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"68146"},"selection_glyph":null,"view":{"id":"68148"}},"id":"68147","type":"GlyphRenderer"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"68158"},"ticker":{"id":"68118"}},"id":"68117","type":"LinearAxis"},{"attributes":{"toolbar":{"id":"68170"},"toolbar_location":"above"},"id":"68171","type":"ToolbarBox"},{"attributes":{"axis":{"id":"68121"},"dimension":1,"ticker":null},"id":"68124","type":"Grid"},{"attributes":{"source":{"id":"68144"}},"id":"68148","type":"CDSView"},{"attributes":{"toolbars":[{"id":"68135"}],"tools":[{"id":"68125"},{"id":"68126"},{"id":"68127"},{"id":"68128"},{"id":"68129"},{"id":"68130"},{"id":"68131"},{"id":"68132"}]},"id":"68170","type":"ProxyToolbar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"68145","type":"Circle"},{"attributes":{},"id":"68130","type":"UndoTool"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAAACamZmZmZmpP5qZmZmZmbk/NDMzMzMzwz+amZmZmZnJPwAAAAAAANA/NDMzMzMz0z9nZmZmZmbWP5qZmZmZmdk/zczMzMzM3D8AAAAAAADgP5qZmZmZmeE/NDMzMzMz4z/NzMzMzMzkP2dmZmZmZuY/AAAAAAAA6D+amZmZmZnpPzQzMzMzM+s/zczMzMzM7D9nZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"PSWUrfrllUC6ZTfE07ObQIHSxpixD6BAXutlA2Fon0AWemawUmmdQBJdptJ4v59A48F0NdPPm0BhLg6FQP6cQNT8SMHKBJ1AN/hQuCybnECYCNGlG7+ZQOL2UqcslZtAhDgC8IHHmkDe0e+ORHOcQJp7JDyiB6BAxVCwvOY1n0BSmX0dHQ6gQFTe5JuCLJtAHVMIAw/Em0COFXr8JyGYQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"68162"},"selection_policy":{"id":"68163"}},"id":"68144","type":"ColumnDataSource"},{"attributes":{},"id":"68128","type":"WheelZoomTool"},{"attributes":{},"id":"68162","type":"Selection"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"68154","type":"Span"},{"attributes":{"callback":null},"id":"68132","type":"HoverTool"},{"attributes":{},"id":"68164","type":"Selection"},{"attributes":{"children":[{"id":"68171"},{"id":"68169"}]},"id":"68172","type":"Column"},{"attributes":{"overlay":{"id":"68134"}},"id":"68129","type":"LassoSelectTool"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"68153","type":"Span"},{"attributes":{},"id":"68118","type":"BasicTicker"},{"attributes":{"source":{"id":"68150"}},"id":"68152","type":"CDSView"},{"attributes":{},"id":"68109","type":"DataRange1d"},{"attributes":{},"id":"68111","type":"DataRange1d"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"68146","type":"Circle"},{"attributes":{"data":{"rug_x":{"__ndarray__":"u/QnuP7V5z9SIsxa1SLgPw==","dtype":"float64","order":"little","shape":[2]},"rug_y":{"__ndarray__":"nB0+wbWyacCcHT7BtbJpwA==","dtype":"float64","order":"little","shape":[2]}},"selected":{"id":"68164"},"selection_policy":{"id":"68165"}},"id":"68150","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"68117"}],"center":[{"id":"68120"},{"id":"68124"}],"left":[{"id":"68121"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"68147"},{"id":"68151"},{"id":"68153"},{"id":"68154"}],"title":{"id":"68155"},"toolbar":{"id":"68135"},"toolbar_location":null,"x_range":{"id":"68109"},"x_scale":{"id":"68113"},"y_range":{"id":"68111"},"y_scale":{"id":"68115"}},"id":"68108","subtype":"Figure","type":"Plot"},{"attributes":{"axis":{"id":"68117"},"ticker":null},"id":"68120","type":"Grid"},{"attributes":{},"id":"68160","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"68133"}},"id":"68127","type":"BoxZoomTool"},{"attributes":{},"id":"68163","type":"UnionRenderers"},{"attributes":{},"id":"68131","type":"SaveTool"},{"attributes":{"text":"mu"},"id":"68155","type":"Title"},{"attributes":{},"id":"68115","type":"LinearScale"},{"attributes":{},"id":"68113","type":"LinearScale"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"68125"},{"id":"68126"},{"id":"68127"},{"id":"68128"},{"id":"68129"},{"id":"68130"},{"id":"68131"},{"id":"68132"}]},"id":"68135","type":"Toolbar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"68133","type":"BoxAnnotation"},{"attributes":{"children":[[{"id":"68108"},0,0]]},"id":"68169","type":"GridBox"},{"attributes":{},"id":"68126","type":"PanTool"},{"attributes":{},"id":"68158","type":"BasicTickFormatter"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"68149","type":"Dash"},{"attributes":{"axis_label":"ESS for small intervals","formatter":{"id":"68160"},"ticker":{"id":"68122"}},"id":"68121","type":"LinearAxis"}],"root_ids":["68172"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"52e1ec3a-f208-440e-9d9d-023e6b226d4e","root_ids":["68172"],"roots":{"68172":"ab4244ec-d320-4b71-9ba7-d9983f1b1ca6"}}];
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