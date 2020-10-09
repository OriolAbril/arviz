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
    
      
      
    
      var element = document.getElementById("06c993c6-7506-4388-ab57-ebdf4018f094");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '06c993c6-7506-4388-ab57-ebdf4018f094' but no matching script tag was found.")
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
                    
                  var docs_json = '{"a7ccc1af-a850-4acb-90e9-75aaebd99970":{"roots":{"references":[{"attributes":{},"id":"4388","type":"BasicTickFormatter"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"4379","type":"Dash"},{"attributes":{"children":[{"id":"4401"},{"id":"4399"}]},"id":"4402","type":"Column"},{"attributes":{},"id":"4341","type":"DataRange1d"},{"attributes":{},"id":"4355","type":"ResetTool"},{"attributes":{"overlay":{"id":"4363"}},"id":"4357","type":"BoxZoomTool"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"4384","type":"Span"},{"attributes":{},"id":"4343","type":"LinearScale"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4375","type":"Circle"},{"attributes":{},"id":"4358","type":"WheelZoomTool"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"4390"},"ticker":{"id":"4348"}},"id":"4347","type":"LinearAxis"},{"attributes":{},"id":"4397","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4364","type":"PolyAnnotation"},{"attributes":{},"id":"4352","type":"BasicTicker"},{"attributes":{},"id":"4348","type":"BasicTicker"},{"attributes":{},"id":"4360","type":"UndoTool"},{"attributes":{"data_source":{"id":"4374"},"glyph":{"id":"4375"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4376"},"selection_glyph":null,"view":{"id":"4378"}},"id":"4377","type":"GlyphRenderer"},{"attributes":{},"id":"4394","type":"UnionRenderers"},{"attributes":{"axis_label":"ESS for small intervals","formatter":{"id":"4388"},"ticker":{"id":"4352"}},"id":"4351","type":"LinearAxis"},{"attributes":{"overlay":{"id":"4364"}},"id":"4359","type":"LassoSelectTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4363","type":"BoxAnnotation"},{"attributes":{"data":{"rug_x":{"__ndarray__":"u/QnuP7V5z9SIsxa1SLgPw==","dtype":"float64","order":"little","shape":[2]},"rug_y":{"__ndarray__":"nB0+wbWyacCcHT7BtbJpwA==","dtype":"float64","order":"little","shape":[2]}},"selected":{"id":"4397"},"selection_policy":{"id":"4396"}},"id":"4380","type":"ColumnDataSource"},{"attributes":{"children":[[{"id":"4338"},0,0]]},"id":"4399","type":"GridBox"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4355"},{"id":"4356"},{"id":"4357"},{"id":"4358"},{"id":"4359"},{"id":"4360"},{"id":"4361"},{"id":"4362"}]},"id":"4365","type":"Toolbar"},{"attributes":{},"id":"4396","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4376","type":"Circle"},{"attributes":{"axis":{"id":"4347"},"ticker":null},"id":"4350","type":"Grid"},{"attributes":{"data_source":{"id":"4380"},"glyph":{"id":"4379"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"4382"}},"id":"4381","type":"GlyphRenderer"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAAACamZmZmZmpP5qZmZmZmbk/NDMzMzMzwz+amZmZmZnJPwAAAAAAANA/NDMzMzMz0z9nZmZmZmbWP5qZmZmZmdk/zczMzMzM3D8AAAAAAADgP5qZmZmZmeE/NDMzMzMz4z/NzMzMzMzkP2dmZmZmZuY/AAAAAAAA6D+amZmZmZnpPzQzMzMzM+s/zczMzMzM7D9nZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"PSWUrfrllUC6ZTfE07ObQIHSxpixD6BAXutlA2Fon0AWemawUmmdQBJdptJ4v59A48F0NdPPm0BhLg6FQP6cQNT8SMHKBJ1AN/hQuCybnECYCNGlG7+ZQOL2UqcslZtAhDgC8IHHmkDe0e+ORHOcQJp7JDyiB6BAxVCwvOY1n0BSmX0dHQ6gQFTe5JuCLJtAHVMIAw/Em0COFXr8JyGYQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4395"},"selection_policy":{"id":"4394"}},"id":"4374","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"4351"},"dimension":1,"ticker":null},"id":"4354","type":"Grid"},{"attributes":{"toolbars":[{"id":"4365"}],"tools":[{"id":"4355"},{"id":"4356"},{"id":"4357"},{"id":"4358"},{"id":"4359"},{"id":"4360"},{"id":"4361"},{"id":"4362"}]},"id":"4400","type":"ProxyToolbar"},{"attributes":{},"id":"4390","type":"BasicTickFormatter"},{"attributes":{},"id":"4356","type":"PanTool"},{"attributes":{"below":[{"id":"4347"}],"center":[{"id":"4350"},{"id":"4354"}],"left":[{"id":"4351"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"4377"},{"id":"4381"},{"id":"4383"},{"id":"4384"}],"title":{"id":"4385"},"toolbar":{"id":"4365"},"toolbar_location":null,"x_range":{"id":"4339"},"x_scale":{"id":"4343"},"y_range":{"id":"4341"},"y_scale":{"id":"4345"}},"id":"4338","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"4361","type":"SaveTool"},{"attributes":{},"id":"4345","type":"LinearScale"},{"attributes":{"toolbar":{"id":"4400"},"toolbar_location":"above"},"id":"4401","type":"ToolbarBox"},{"attributes":{"source":{"id":"4374"}},"id":"4378","type":"CDSView"},{"attributes":{"text":"mu"},"id":"4385","type":"Title"},{"attributes":{"source":{"id":"4380"}},"id":"4382","type":"CDSView"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"4383","type":"Span"},{"attributes":{},"id":"4339","type":"DataRange1d"},{"attributes":{},"id":"4395","type":"Selection"},{"attributes":{"callback":null},"id":"4362","type":"HoverTool"}],"root_ids":["4402"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"a7ccc1af-a850-4acb-90e9-75aaebd99970","root_ids":["4402"],"roots":{"4402":"06c993c6-7506-4388-ab57-ebdf4018f094"}}];
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