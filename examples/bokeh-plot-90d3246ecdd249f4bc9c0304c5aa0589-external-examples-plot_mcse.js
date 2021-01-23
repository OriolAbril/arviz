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
    
      
      
    
      var element = document.getElementById("7f5ea1b8-c7f7-4d9f-ab34-61b9a80668d6");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '7f5ea1b8-c7f7-4d9f-ab34-61b9a80668d6' but no matching script tag was found.")
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
                    
                  var docs_json = '{"e93b1cd8-3c77-4fb5-9aad-0dbaca90478e":{"roots":{"references":[{"attributes":{"text":"mu"},"id":"35096","type":"Title"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"35118"},"selection_policy":{"id":"35117"}},"id":"35084","type":"ColumnDataSource"},{"attributes":{},"id":"35008","type":"BasicTicker"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"35089","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"35090","type":"Span"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"35078","type":"Dash"},{"attributes":{},"id":"35114","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"35023"}},"id":"35017","type":"BoxZoomTool"},{"attributes":{},"id":"35021","type":"SaveTool"},{"attributes":{"data_source":{"id":"35093"},"glyph":{"id":"35092"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"35095"}},"id":"35094","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"35091","type":"Span"},{"attributes":{},"id":"35020","type":"UndoTool"},{"attributes":{},"id":"35102","type":"BasicTickFormatter"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"35092","type":"Dash"},{"attributes":{"axis":{"id":"35007"},"ticker":null},"id":"35010","type":"Grid"},{"attributes":{"callback":null},"id":"35022","type":"HoverTool"},{"attributes":{"data_source":{"id":"35084"},"glyph":{"id":"35085"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"35086"},"selection_glyph":null,"view":{"id":"35088"}},"id":"35087","type":"GlyphRenderer"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"35108"},"selection_policy":{"id":"35107"}},"id":"35079","type":"ColumnDataSource"},{"attributes":{"text":"tau"},"id":"35082","type":"Title"},{"attributes":{"source":{"id":"35093"}},"id":"35095","type":"CDSView"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"35076","type":"Span"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"35059","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"35077","type":"Span"},{"attributes":{"data_source":{"id":"35079"},"glyph":{"id":"35078"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"35081"}},"id":"35080","type":"GlyphRenderer"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"35060","type":"PolyAnnotation"},{"attributes":{"source":{"id":"35070"}},"id":"35074","type":"CDSView"},{"attributes":{"end":1,"start":-0.05},"id":"35001","type":"DataRange1d"},{"attributes":{"data_source":{"id":"35070"},"glyph":{"id":"35071"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"35072"},"selection_glyph":null,"view":{"id":"35074"}},"id":"35073","type":"GlyphRenderer"},{"attributes":{"toolbars":[{"id":"35025"},{"id":"35061"}],"tools":[{"id":"35015"},{"id":"35016"},{"id":"35017"},{"id":"35018"},{"id":"35019"},{"id":"35020"},{"id":"35021"},{"id":"35022"},{"id":"35051"},{"id":"35052"},{"id":"35053"},{"id":"35054"},{"id":"35055"},{"id":"35056"},{"id":"35057"},{"id":"35058"}]},"id":"35123","type":"ProxyToolbar"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"35075","type":"Span"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"35023","type":"BoxAnnotation"},{"attributes":{},"id":"35117","type":"UnionRenderers"},{"attributes":{},"id":"34999","type":"DataRange1d"},{"attributes":{},"id":"35118","type":"Selection"},{"attributes":{},"id":"35105","type":"UnionRenderers"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"35024","type":"PolyAnnotation"},{"attributes":{},"id":"35106","type":"Selection"},{"attributes":{},"id":"35003","type":"LinearScale"},{"attributes":{},"id":"35119","type":"UnionRenderers"},{"attributes":{},"id":"35120","type":"Selection"},{"attributes":{},"id":"35005","type":"LinearScale"},{"attributes":{},"id":"35107","type":"UnionRenderers"},{"attributes":{"toolbar":{"id":"35123"},"toolbar_location":"above"},"id":"35124","type":"ToolbarBox"},{"attributes":{},"id":"35108","type":"Selection"},{"attributes":{"children":[[{"id":"34998"},0,0],[{"id":"35034"},0,1]]},"id":"35122","type":"GridBox"},{"attributes":{"below":[{"id":"35043"}],"center":[{"id":"35046"},{"id":"35050"}],"left":[{"id":"35047"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"35087"},{"id":"35089"},{"id":"35090"},{"id":"35091"},{"id":"35094"}],"title":{"id":"35096"},"toolbar":{"id":"35061"},"toolbar_location":null,"x_range":{"id":"35035"},"x_scale":{"id":"35039"},"y_range":{"id":"35037"},"y_scale":{"id":"35041"}},"id":"35034","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"35120"},"selection_policy":{"id":"35119"}},"id":"35093","type":"ColumnDataSource"},{"attributes":{},"id":"35018","type":"WheelZoomTool"},{"attributes":{"source":{"id":"35084"}},"id":"35088","type":"CDSView"},{"attributes":{},"id":"35056","type":"UndoTool"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"35100"},"ticker":{"id":"35008"}},"id":"35007","type":"LinearAxis"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"35086","type":"Circle"},{"attributes":{},"id":"35015","type":"ResetTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"35015"},{"id":"35016"},{"id":"35017"},{"id":"35018"},{"id":"35019"},{"id":"35020"},{"id":"35021"},{"id":"35022"}]},"id":"35025","type":"Toolbar"},{"attributes":{},"id":"35035","type":"DataRange1d"},{"attributes":{"children":[{"id":"35124"},{"id":"35122"}]},"id":"35125","type":"Column"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"35112"},"ticker":{"id":"35044"}},"id":"35043","type":"LinearAxis"},{"attributes":{"end":1,"start":-0.05},"id":"35037","type":"DataRange1d"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"35114"},"ticker":{"id":"35048"}},"id":"35047","type":"LinearAxis"},{"attributes":{"axis":{"id":"35011"},"dimension":1,"ticker":null},"id":"35014","type":"Grid"},{"attributes":{},"id":"35039","type":"LinearScale"},{"attributes":{"overlay":{"id":"35024"}},"id":"35019","type":"LassoSelectTool"},{"attributes":{},"id":"35041","type":"LinearScale"},{"attributes":{},"id":"35112","type":"BasicTickFormatter"},{"attributes":{"callback":null},"id":"35058","type":"HoverTool"},{"attributes":{},"id":"35044","type":"BasicTicker"},{"attributes":{"axis":{"id":"35043"},"ticker":null},"id":"35046","type":"Grid"},{"attributes":{},"id":"35012","type":"BasicTicker"},{"attributes":{"below":[{"id":"35007"}],"center":[{"id":"35010"},{"id":"35014"}],"left":[{"id":"35011"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"35073"},{"id":"35075"},{"id":"35076"},{"id":"35077"},{"id":"35080"}],"title":{"id":"35082"},"toolbar":{"id":"35025"},"toolbar_location":null,"x_range":{"id":"34999"},"x_scale":{"id":"35003"},"y_range":{"id":"35001"},"y_scale":{"id":"35005"}},"id":"34998","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"35100","type":"BasicTickFormatter"},{"attributes":{},"id":"35016","type":"PanTool"},{"attributes":{"axis":{"id":"35047"},"dimension":1,"ticker":null},"id":"35050","type":"Grid"},{"attributes":{},"id":"35048","type":"BasicTicker"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"35106"},"selection_policy":{"id":"35105"}},"id":"35070","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"35085","type":"Circle"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"35072","type":"Circle"},{"attributes":{"overlay":{"id":"35059"}},"id":"35053","type":"BoxZoomTool"},{"attributes":{},"id":"35052","type":"PanTool"},{"attributes":{},"id":"35051","type":"ResetTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"35071","type":"Circle"},{"attributes":{},"id":"35057","type":"SaveTool"},{"attributes":{},"id":"35054","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"35060"}},"id":"35055","type":"LassoSelectTool"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"35102"},"ticker":{"id":"35012"}},"id":"35011","type":"LinearAxis"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"35051"},{"id":"35052"},{"id":"35053"},{"id":"35054"},{"id":"35055"},{"id":"35056"},{"id":"35057"},{"id":"35058"}]},"id":"35061","type":"Toolbar"},{"attributes":{"source":{"id":"35079"}},"id":"35081","type":"CDSView"}],"root_ids":["35125"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"e93b1cd8-3c77-4fb5-9aad-0dbaca90478e","root_ids":["35125"],"roots":{"35125":"7f5ea1b8-c7f7-4d9f-ab34-61b9a80668d6"}}];
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