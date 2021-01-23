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
    
      
      
    
      var element = document.getElementById("258f095b-d1cd-4dac-8b42-001c777cac3b");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '258f095b-d1cd-4dac-8b42-001c777cac3b' but no matching script tag was found.")
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
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
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
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
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
                    
                  var docs_json = '{"5b606074-b9f2-40e5-9b33-72151ca994a0":{"roots":{"references":[{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"4271","type":"BoxAnnotation"},{"attributes":{},"id":"4251","type":"LinearScale"},{"attributes":{"source":{"id":"4288"}},"id":"4290","type":"CDSView"},{"attributes":{"line_color":"red","line_dash":[6],"line_width":3,"location":400},"id":"4292","type":"Span"},{"attributes":{"source":{"id":"4282"}},"id":"4286","type":"CDSView"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"4272","type":"PolyAnnotation"},{"attributes":{},"id":"4298","type":"BasicTickFormatter"},{"attributes":{"children":[[{"id":"4246"},0,0]]},"id":"4307","type":"GridBox"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"4291","type":"Span"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"4296"},"ticker":{"id":"4256"}},"id":"4255","type":"LinearAxis"},{"attributes":{},"id":"4300","type":"Selection"},{"attributes":{"axis_label":"ESS for small intervals","formatter":{"id":"4298"},"ticker":{"id":"4260"}},"id":"4259","type":"LinearAxis"},{"attributes":{},"id":"4301","type":"UnionRenderers"},{"attributes":{"toolbars":[{"id":"4273"}],"tools":[{"id":"4263"},{"id":"4264"},{"id":"4265"},{"id":"4266"},{"id":"4267"},{"id":"4268"},{"id":"4269"},{"id":"4270"}]},"id":"4308","type":"ProxyToolbar"},{"attributes":{},"id":"4256","type":"BasicTicker"},{"attributes":{"callback":null},"id":"4270","type":"HoverTool"},{"attributes":{},"id":"4253","type":"LinearScale"},{"attributes":{"text":"mu"},"id":"4293","type":"Title"},{"attributes":{"axis":{"id":"4255"},"ticker":null},"id":"4258","type":"Grid"},{"attributes":{"toolbar":{"id":"4308"},"toolbar_location":"above"},"id":"4309","type":"ToolbarBox"},{"attributes":{"axis":{"id":"4259"},"dimension":1,"ticker":null},"id":"4262","type":"Grid"},{"attributes":{},"id":"4260","type":"BasicTicker"},{"attributes":{},"id":"4302","type":"Selection"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4284","type":"Circle"},{"attributes":{"overlay":{"id":"4271"}},"id":"4265","type":"BoxZoomTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":6},"x":{"field":"x"},"y":{"field":"y"}},"id":"4283","type":"Circle"},{"attributes":{},"id":"4303","type":"UnionRenderers"},{"attributes":{},"id":"4264","type":"PanTool"},{"attributes":{},"id":"4263","type":"ResetTool"},{"attributes":{},"id":"4269","type":"SaveTool"},{"attributes":{"below":[{"id":"4255"}],"center":[{"id":"4258"},{"id":"4262"}],"left":[{"id":"4259"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"4285"},{"id":"4289"},{"id":"4291"},{"id":"4292"}],"title":{"id":"4293"},"toolbar":{"id":"4273"},"toolbar_location":null,"x_range":{"id":"4247"},"x_scale":{"id":"4251"},"y_range":{"id":"4249"},"y_scale":{"id":"4253"}},"id":"4246","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"4266","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"4272"}},"id":"4267","type":"LassoSelectTool"},{"attributes":{},"id":"4268","type":"UndoTool"},{"attributes":{},"id":"4247","type":"DataRange1d"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"4287","type":"Dash"},{"attributes":{},"id":"4296","type":"BasicTickFormatter"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"4263"},{"id":"4264"},{"id":"4265"},{"id":"4266"},{"id":"4267"},{"id":"4268"},{"id":"4269"},{"id":"4270"}]},"id":"4273","type":"Toolbar"},{"attributes":{"data":{"x":{"__ndarray__":"AAAAAAAAAACamZmZmZmpP5qZmZmZmbk/NDMzMzMzwz+amZmZmZnJPwAAAAAAANA/NDMzMzMz0z9nZmZmZmbWP5qZmZmZmdk/zczMzMzM3D8AAAAAAADgP5qZmZmZmeE/NDMzMzMz4z/NzMzMzMzkP2dmZmZmZuY/AAAAAAAA6D+amZmZmZnpPzQzMzMzM+s/zczMzMzM7D9nZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"PSWUrfrllUC6ZTfE07ObQIHSxpixD6BAXutlA2Fon0AWemawUmmdQBJdptJ4v59A48F0NdPPm0BhLg6FQP6cQNT8SMHKBJ1AN/hQuCybnECYCNGlG7+ZQOL2UqcslZtAhDgC8IHHmkDe0e+ORHOcQJp7JDyiB6BAxVCwvOY1n0BSmX0dHQ6gQFTe5JuCLJtAHVMIAw/Em0COFXr8JyGYQA==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"4300"},"selection_policy":{"id":"4301"}},"id":"4282","type":"ColumnDataSource"},{"attributes":{"children":[{"id":"4309"},{"id":"4307"}]},"id":"4310","type":"Column"},{"attributes":{"data":{"rug_x":{"__ndarray__":"u/QnuP7V5z9SIsxa1SLgPw==","dtype":"float64","order":"little","shape":[2]},"rug_y":{"__ndarray__":"nB0+wbWyacCcHT7BtbJpwA==","dtype":"float64","order":"little","shape":[2]}},"selected":{"id":"4302"},"selection_policy":{"id":"4303"}},"id":"4288","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4282"},"glyph":{"id":"4283"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"4284"},"selection_glyph":null,"view":{"id":"4286"}},"id":"4285","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"4288"},"glyph":{"id":"4287"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"4290"}},"id":"4289","type":"GlyphRenderer"},{"attributes":{},"id":"4249","type":"DataRange1d"}],"root_ids":["4310"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"5b606074-b9f2-40e5-9b33-72151ca994a0","root_ids":["4310"],"roots":{"4310":"258f095b-d1cd-4dac-8b42-001c777cac3b"}}];
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